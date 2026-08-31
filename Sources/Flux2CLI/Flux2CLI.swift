// Flux2CLI.swift - Command Line Interface for Flux.2
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import ImageIO
import UniformTypeIdentifiers

/// Configure custom models directory for both registries
func configureModelsDirectory(_ path: String?) {
    guard let path = path else { return }
    let url = URL(fileURLWithPath: path)
    ModelRegistry.customModelsDirectory = url
    TextEncoderModelDownloader.customModelsDirectory = url
    TextEncoderModelDownloader.reconfigureHubApi()
}

/// Parse repeatable CLI LoRA arguments.
///
/// A file spec may optionally end in `:scale`, for example
/// `/models/style.safetensors:0.7`. A path without an inline scale uses the
/// command's `--lora-scale` value. JSON configs keep their own scale and
/// metadata, so they are handled separately below.
private func parseLoRAFileSpec(_ spec: String, defaultScale: Float) throws -> LoRAConfig {
    var path = spec
    var scale = defaultScale

    if let separator = spec.lastIndex(of: ":"), separator != spec.startIndex {
        let scaleText = String(spec[spec.index(after: separator)...])
        guard let inlineScale = Float(scaleText), inlineScale.isFinite else {
            throw ValidationError("Invalid LoRA scale in '\(spec)'. Use PATH or PATH:SCALE, e.g. style.safetensors:0.7")
        }
        path = String(spec[..<separator])
        scale = inlineScale
    }

    guard !path.isEmpty else {
        throw ValidationError("LoRA path cannot be empty")
    }
    guard FileManager.default.fileExists(atPath: path) else {
        throw ValidationError("LoRA file not found: \(path)")
    }

    return LoRAConfig(filePath: path, scale: scale)
}

/// Build all LoRA configs for a generation command. Raw files and JSON
/// configs are intentionally mutually exclusive so scale and scheduler
/// precedence remain unambiguous.
private func loadLoRAConfigs(
    fileSpecs: [String],
    configPaths: [String],
    defaultScale: Float
) throws -> [LoRAConfig] {
    guard fileSpecs.isEmpty || configPaths.isEmpty else {
        throw ValidationError("Use either --lora or --lora-config, not both")
    }

    if !configPaths.isEmpty {
        return try configPaths.map { configPath in
            guard FileManager.default.fileExists(atPath: configPath) else {
                throw ValidationError("LoRA config file not found: \(configPath)")
            }
            do {
                let config = try LoRAConfig.load(from: configPath)
                guard FileManager.default.fileExists(atPath: config.filePath) else {
                    throw ValidationError("LoRA file specified in config not found: \(config.filePath)")
                }
                return config
            } catch let error as ValidationError {
                throw error
            } catch let error as DecodingError {
                throw ValidationError("Invalid LoRA config JSON (\(configPath)): \(error)")
            } catch {
                throw ValidationError("Failed to load LoRA config \(configPath): \(error.localizedDescription)")
            }
        }
    }

    return try fileSpecs.map { try parseLoRAFileSpec($0, defaultScale: defaultScale) }
}

/// Validate the negative-prompt combination before loading large models.
/// Classical CFG is only meaningful for the non-distilled Klein base
/// checkpoints; distilled Klein and Dev use different guidance mechanisms.
func validateNegativePrompt(
    _ negativePrompt: String?,
    model: Flux2Model,
    guidance: Float
) throws {
    guard let negativePrompt,
          !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return
    }
    guard model.usesClassicalCFG else {
        throw ValidationError(
            "--negative-prompt requires a non-distilled Klein base model (klein-4b-base or klein-9b-base); \(model.displayName) does not support classical negative-prompt CFG")
    }
    guard guidance > 1.0 else {
        throw ValidationError(
            "--negative-prompt requires --guidance greater than 1.0 so classifier-free guidance is active (received \(guidance))")
    }
}

func parsePromptUpsampler(
    _ value: String,
    path: String?
) throws -> (model: Flux2PromptUpsampler, path: URL?) {
    let selected = try Flux2PromptUpsampler.parseCLI(value)
    guard let path else {
        return (selected, nil)
    }

    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw ValidationError("Prompt upsampler path is not a directory: \(path)")
    }
    return (selected, URL(fileURLWithPath: path))
}

@main
struct Flux2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flux2",
        abstract: "Flux.2 image generation on Mac with MLX",
        version: Flux2Core.version,
        subcommands: [
            TextToImage.self,
            ImageToImage.self,
            Inpaint.self,
            Outpaint.self,
            MaskSubject.self,
            Download.self,
            ExportQuantized.self,
            Info.self,
            Profile.self,
            VLMTest.self,
            CompareEncoders.self,
            TestVLGeneration.self,
            TestQwen35.self,
            TestGemma4.self,
            EvaluateLoRA.self,
            TrainLoRA.self,
            TrainingControlCommand.self,
        ],
        defaultSubcommand: TextToImage.self
    )
}

// MARK: - Text-to-Image Command

struct TextToImage: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "t2i",
        abstract: "Generate image from text prompt"
    )

    @Argument(help: "Text prompt for image generation")
    var prompt: String

    @Option(name: .long, help: "Negative prompt for classical CFG (requires klein-4b-base or klein-9b-base)")
    var negativePrompt: String?

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String = "output.png"

    @Option(name: .shortAndLong, help: "Image width")
    var width: Int = 1024

    @Option(name: .shortAndLong, help: "Image height")
    var height: Int = 1024

    @Option(name: .shortAndLong, help: "Number of inference steps (default: 28 for Dev, 4 for Klein)")
    var steps: Int?

    @Option(name: .shortAndLong, help: "Guidance scale (default: 4.0 for Dev, 1.0 for Klein)")
    var guidance: Float?

    @Option(name: .long, help: "Random seed")
    var seed: UInt64?

    @Option(name: .long, help: "Model variant: dev (32B), klein-4b (4B, Apache 2.0), klein-9b (9B), klein-9b-kv (9B, KV-cached I2I)")
    var model: String = "dev"

    @Option(name: .long, help: "Text encoder quantization: bf16, 8bit, 6bit, 4bit")
    var textQuant: String = "8bit"

    @Option(name: .long, help: "Transformer quantization: \(TransformerQuantization.cliValueList)")
    var transformerQuant: String = "qint8"

    @Flag(name: .long, help: "Show detailed logs (model loading, config, VLM interpretation)")
    var verbose: Bool = false

    @Flag(name: .long, help: "Enable performance profiling")
    var profile: Bool = false

    @OptionGroup var beaconOptions: BeaconOptions

    @Flag(name: .long, help: "Enhance prompt with more visual details before encoding")
    var upsamplePrompt: Bool = false

    @Option(name: .long, help: "Prompt upsampler model: auto (default), mistral, or qwen3")
    var upsampleModel: String = "auto"

    @Option(name: .long, help: "Local model directory for --upsample-model (Mistral or Qwen3)")
    var upsampleModelPath: String?

    @Option(name: .long, help: "Image to analyze with VLM and inject description into prompt (semantic interpretation)")
    var interpret: [String] = []

    @Option(name: .long, help: "Save intermediate images at each N steps (e.g., 5 saves every 5 steps)")
    var checkpoint: Int?

    @Option(name: .long, help: "LoRA adapter file; repeat for multiple adapters, optionally append :SCALE")
    var lora: [String] = []

    @Option(name: .long, help: "Default LoRA scale for --lora specs without an inline :SCALE (default: 1.0)")
    var loraScale: Float = 1.0

    @Option(name: .long, help: "LoRA config JSON file; repeat for multiple adapters (alternative to --lora)")
    var loraConfig: [String] = []

    @Option(name: .long, help: "VAE variant: small-decoder (default, distilled, ~1.4x faster), standard")
    var vaeVariant: String = "small-decoder"

    @Option(name: .long, help: "Memory profile: auto (default), conservative, balanced, performance")
    var memoryProfile: String = "auto"

    @Option(name: .long, help: "HuggingFace token for gated models (or set HF_TOKEN env var)")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage)")
    var modelsDir: String?

    func run() async throws {
        // Configure custom models directory
        configureModelsDirectory(modelsDir)

        beaconOptions.activate()

        // Configure logging verbosity
        if verbose {
            Flux2Debug.enableDebugMode()
            FluxDebug.isEnabled = true
        } else {
            Flux2Debug.setNormalMode()
            FluxDebug.isEnabled = false
        }

        // Configure profiling
        if profile {
            Flux2Profiler.shared.enable()
        }
        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, or klein-9b")
        }
        let promptUpsampler = try parsePromptUpsampler(upsampleModel, path: upsampleModelPath)

        // Check if model requires authentication (gated models)
        // Gated: dev, klein-9b (both base and distilled)
        // Non-gated: klein-4b
        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
        let isGatedModel = modelVariant == .dev || modelVariant == .klein9B || modelVariant == .klein9BKV

        if token == nil && isGatedModel {
            print("⚠️  No HuggingFace token provided.")
            print("   \(modelVariant.displayName) is a gated model. You need to:")
            print("   1. Accept the license at https://huggingface.co/black-forest-labs/\(modelVariant == .dev ? "FLUX.2-dev" : "FLUX.2-klein-9B")")
            print("   2. Set HF_TOKEN environment variable or use --hf-token")
            print()
        }

        // Load all LoRAs early so scheduler overrides are available before generation.
        let loraConfigs = try loadLoRAConfigs(
            fileSpecs: lora,
            configPaths: loraConfig,
            defaultScale: loraScale
        )

        // Apply model-specific defaults for Klein (distilled for 4 steps, guidance 1.0)
        // LoRA scheduler overrides take precedence if not explicitly set by user.
        // If several configs contain overrides, the last one wins, matching the
        // pipeline's activeSchedulerOverrides behavior.
        let actualSteps: Int
        let actualGuidance: Float
        let loraOverrides = loraConfigs.compactMap { $0.schedulerOverrides }.last

        // Priority: CLI flag > LoRA override > model default
        actualSteps = steps ?? loraOverrides?.numSteps ?? modelVariant.defaultSteps
        actualGuidance = guidance ?? loraOverrides?.guidance ?? modelVariant.defaultGuidance
        try validateNegativePrompt(negativePrompt, model: modelVariant, guidance: actualGuidance)

        // Parse quantization settings
        guard let textQuantization = MistralQuantization(rawValue: textQuant) else {
            throw ValidationError("Invalid text quantization: \(textQuant). Use bf16, 8bit, 6bit, or 4bit")
        }

        let transformerQuantization = try TransformerQuantization.parseCLI(transformerQuant)

        let quantConfig = Flux2QuantizationConfig(
            textEncoder: textQuantization,
            transformer: transformerQuantization
        )

        // Warn if bf16 with insufficient RAM
        if transformerQuantization == .bf16 {
            let systemRAM = ModelRegistry.systemRAMGB
            if systemRAM < 96 {
                print("⚠️  Warning: bf16 transformer requires ~90GB RAM, you have \(systemRAM)GB")
                print("   Consider using --transformer-quant qint8 for lower memory usage")
                print()
            }
        }

        if verbose {
            print("Configuration:")
            print("  Model: \(modelVariant.displayName)")
            print("  Text encoder: \(textQuantization.displayName)")
            print("  Transformer: \(transformerQuantization.displayName)")
            print("  Estimated memory: ~\(quantConfig.estimatedTotalMemoryGB)GB")
            print()
        }

        // Validate interpret image paths exist
        var interpretImagePaths: [String] = []
        for path in interpret {
            guard FileManager.default.fileExists(atPath: path) else {
                throw ValidationError("Interpret image not found: \(path)")
            }
            interpretImagePaths.append(path)
        }

        print("Generating \(width)x\(height), \(actualSteps) steps, guidance \(actualGuidance)\(seed.map { ", seed \($0)" } ?? "")...")
        if verbose {
            print("  Prompt: \"\(prompt)\"")
            if let negativePrompt, !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("  Negative prompt: \"\(negativePrompt)\"")
            }
            if !interpretImagePaths.isEmpty {
                print("  Interpret images: \(interpretImagePaths.count) (VLM will analyze and enrich prompt)")
                for path in interpretImagePaths {
                    print("    - \(path)")
                }
            }
            if upsamplePrompt {
                print("  Prompt upsampling: enabled (will enhance prompt with visual details)")
                print("  Prompt upsampler: \(promptUpsampler.model.displayName)\(promptUpsampler.path.map { " [\($0.path)]" } ?? "")")
            }
            if !loraConfigs.isEmpty {
                print("  LoRAs: \(loraConfigs.count)")
                for config in loraConfigs {
                    print("    - \(config.name) (scale: \(config.effectiveScale))")
                    if let overrides = config.schedulerOverrides, overrides.hasOverrides {
                        if overrides.customSigmas != nil {
                            print("      - Custom sigmas: \(overrides.customSigmas!.count) values")
                        }
                        if let recSteps = overrides.numSteps, steps == nil {
                            print("      - Using recommended steps: \(recSteps)")
                        }
                        if let recGuidance = overrides.guidance, guidance == nil {
                            print("      - Using recommended guidance: \(recGuidance)")
                        }
                    }
                }
            }
            if let checkpointInterval = checkpoint {
                print("  Checkpoints: every \(checkpointInterval) step(s)")
            }
            print()
        }

        // Parse VAE variant
        guard let vaeVar = ModelRegistry.VAEVariant(rawValue: vaeVariant) else {
            throw ValidationError("Invalid VAE variant: \(vaeVariant). Use standard or small-decoder")
        }

        // Create pipeline with HuggingFace token
        let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig, vaeVariant: vaeVar, hfToken: token)
        pipeline.promptUpsampler = promptUpsampler.model
        pipeline.promptUpsamplerPath = promptUpsampler.path

        // Set memory profile
        switch memoryProfile.lowercased() {
        case "conservative": pipeline.memoryProfile = .conservative
        case "balanced": pipeline.memoryProfile = .balanced
        case "performance": pipeline.memoryProfile = .performance
        case "auto": pipeline.memoryProfile = .auto
        default:
            throw ValidationError("Invalid memory profile: \(memoryProfile). Use auto, conservative, balanced, or performance")
        }

        // Load all LoRAs before the transformer is loaded so they can be fused
        // together in one pass (important for quantized transformers).
        for config in loraConfigs {
            do {
                let info = try pipeline.loadLoRA(config)
                if verbose {
                    print("LoRA loaded: \(info.numLayers) layers, rank \(info.rank), \(String(format: "%.1f", info.memorySizeMB)) MB — \(config.name)")
                }
                if info.targetModel != .unknown && info.targetModel.rawValue != modelVariant.rawValue {
                    print("⚠️  Warning: LoRA \(config.name) was trained for \(info.targetModel.rawValue), but using \(modelVariant.rawValue)")
                }
            } catch {
                throw ValidationError("Failed to load LoRA \(config.name): \(error.localizedDescription)")
            }
        }

        // Check for missing models
        if !pipeline.hasRequiredModels {
            let missing = pipeline.missingModels
            print("Missing models:")
            for m in missing {
                print("  - \(m.displayName)")
            }
            print()
            print("Please download the required models first.")
            throw ExitCode.failure
        }

        // Generate
        let startTime = Date()

        // Prepare checkpoint directory if needed
        let checkpointDir: String?
        if let _ = checkpoint {
            // Create checkpoint directory based on output path
            let outputURL = URL(fileURLWithPath: output)
            let baseName = outputURL.deletingPathExtension().lastPathComponent
            let parentDir = outputURL.deletingLastPathComponent().path
            checkpointDir = "\(parentDir)/\(baseName)_checkpoints"

            // Create directory
            try FileManager.default.createDirectory(
                atPath: checkpointDir!,
                withIntermediateDirectories: true
            )
            if verbose {
                print("Checkpoints will be saved to: \(checkpointDir!)")
            }
        } else {
            checkpointDir = nil
        }

        let image = try await pipeline.generateTextToImage(
            prompt: prompt,
            negativePrompt: negativePrompt,
            interpretImagePaths: interpretImagePaths.isEmpty ? nil : interpretImagePaths,
            height: height,
            width: width,
            steps: actualSteps,
            guidance: actualGuidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpoint
        ) { current, total in
            let progress = Float(current) / Float(total) * 100
            print("\rStep \(current)/\(total) [\(String(format: "%.0f", progress))%]", terminator: "")
            fflush(stdout)
        } onCheckpoint: { step, checkpointImage in
            if let dir = checkpointDir {
                let checkpointPath = "\(dir)/step_\(String(format: "%03d", step)).png"
                do {
                    try saveImage(checkpointImage, to: checkpointPath)
                    print("\n  Checkpoint saved: step_\(String(format: "%03d", step)).png")
                } catch {
                    print("\n  Failed to save checkpoint at step \(step): \(error.localizedDescription)")
                }
            }
        }

        print()

        let elapsed = Date().timeIntervalSince(startTime)
        print("Generation completed in \(String(format: "%.1f", elapsed))s")

        // Save image
        try saveImage(image, to: output)
        print("Image saved to \(output)")
    }
}

// MARK: - Image-to-Image Command

struct ImageToImage: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "i2i",
        abstract: "Transform an image using a text prompt (image-to-image)"
    )

    @Argument(help: "Text prompt describing the desired output")
    var prompt: String

    @Option(name: .long, help: "Negative prompt for classical CFG (requires klein-4b-base or klein-9b-base)")
    var negativePrompt: String?

    @Option(name: .shortAndLong, help: "Reference image for visual conditioning")
    var images: [String]

    @Option(name: .long, help: "Image to analyze with VLM and inject description into prompt (not used as visual reference)")
    var interpret: [String] = []

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String = "output.png"

    @Option(name: .shortAndLong, help: "Number of effective denoising steps (default: 28 for Dev, 4 for Klein)")
    var steps: Int?

@Option(name: .shortAndLong, help: "Guidance scale (default: 4.0 for Dev, 1.0 for Klein)")
    var guidance: Float?

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .shortAndLong, help: "Output image width (default: from first reference image)")
    var width: Int?

    @Option(name: .shortAndLong, help: "Output image height (default: from first reference image)")
    var height: Int?

    @Option(name: .long, help: "Max VAE encode budget per reference image, in megapixels (1 MP = 1024×1024; default 1.0). Raise for higher-fidelity conditioning at the cost of memory; e.g. 4.0 ≈ 2048².")
    var maxReferenceMegapixels: Double?

    @Flag(name: .long, help: "Enhance prompt with visual details using the selected prompt upsampler")
    var upsamplePrompt: Bool = false

    @Option(name: .long, help: "Prompt upsampler model: auto (default), mistral, or qwen3")
    var upsampleModel: String = "auto"

    @Option(name: .long, help: "Local model directory for --upsample-model (Mistral or Qwen3)")
    var upsampleModelPath: String?

    @Option(name: .long, help: "Save checkpoint image every N steps")
    var checkpoint: Int?

    @Flag(name: .long, help: "Show detailed performance profiling")
    var profile: Bool = false

    @OptionGroup var beaconOptions: BeaconOptions

    @Flag(name: .long, help: "Show detailed logs (model loading, config, VLM interpretation)")
    var verbose: Bool = false

    @Option(name: .long, help: "Model variant: dev (32B), klein-4b (4B, Apache 2.0), klein-9b (9B), klein-9b-kv (9B, KV-cached I2I)")
    var model: String = "dev"

    @Option(name: .long, help: "Text encoder quantization: bf16, 8bit, 6bit, 4bit")
    var textQuant: String = "8bit"

    @Option(name: .long, help: "Transformer quantization: \(TransformerQuantization.cliValueList)")
    var transformerQuant: String = "qint8"

    @Option(name: .long, help: "LoRA adapter file; repeat for multiple adapters, optionally append :SCALE")
    var lora: [String] = []

    @Option(name: .long, help: "Default LoRA scale for --lora specs without an inline :SCALE (default: 1.0)")
    var loraScale: Float = 1.0

    @Option(name: .long, help: "LoRA config JSON file; repeat for multiple adapters (alternative to --lora)")
    var loraConfig: [String] = []

    @Option(name: .long, help: "VAE variant: small-decoder (default, distilled, ~1.4x faster), standard")
    var vaeVariant: String = "small-decoder"

    @Option(name: .long, help: "Memory profile: auto (default), conservative, balanced, performance")
    var memoryProfile: String = "auto"

    @Option(name: .long, help: "HuggingFace token for gated models (or set HF_TOKEN env var)")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage)")
    var modelsDir: String?

    func run() async throws {
        let startTime = Date()

        // Configure custom models directory
        configureModelsDirectory(modelsDir)

        beaconOptions.activate()

        // Configure logging verbosity
        if verbose {
            Flux2Debug.enableDebugMode()
            FluxDebug.isEnabled = true
        } else {
            Flux2Debug.setNormalMode()
            FluxDebug.isEnabled = false
        }

        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, or klein-9b")
        }
        let promptUpsampler = try parsePromptUpsampler(upsampleModel, path: upsampleModelPath)

        // Check if model requires authentication (gated models)
        // Gated: dev, klein-9b (both base and distilled)
        // Non-gated: klein-4b
        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
        let isGatedModel = modelVariant == .dev || modelVariant == .klein9B || modelVariant == .klein9BKV

        if token == nil && isGatedModel {
            print("⚠️  No HuggingFace token provided.")
            print("   \(modelVariant.displayName) is a gated model. You need to:")
            print("   1. Accept the license at https://huggingface.co/black-forest-labs/\(modelVariant == .dev ? "FLUX.2-dev" : "FLUX.2-klein-9B")")
            print("   2. Set HF_TOKEN environment variable or use --hf-token")
            print()
        }

        // Load all LoRAs early so scheduler overrides are available before generation.
        let loraConfigs = try loadLoRAConfigs(
            fileSpecs: lora,
            configPaths: loraConfig,
            defaultScale: loraScale
        )

        // Apply model-specific defaults for Klein (distilled for 4 steps, guidance 1.0)
        // LoRA scheduler overrides take precedence if not explicitly set by user.
        let actualSteps: Int
        let actualGuidance: Float
        let loraOverrides = loraConfigs.compactMap { $0.schedulerOverrides }.last

        // Priority: CLI flag > LoRA override > model default
        actualSteps = steps ?? loraOverrides?.numSteps ?? modelVariant.defaultSteps
        actualGuidance = guidance ?? loraOverrides?.guidance ?? modelVariant.defaultGuidance
        try validateNegativePrompt(negativePrompt, model: modelVariant, guidance: actualGuidance)

        // Validate image count (model-specific limit)
        let maxImages = modelVariant.maxReferenceImages
        guard !images.isEmpty && images.count <= maxImages else {
            if images.isEmpty {
                throw ValidationError("Please provide 1-\(maxImages) reference images with --images")
            } else {
                throw ValidationError("Maximum \(maxImages) reference images allowed for \(modelVariant.displayName)")
            }
        }

        // Configure profiling
        if profile {
            Flux2Profiler.shared.enable()
        }

        // Load reference images (visual conditioning)
        var refImages: [CGImage] = []
        for path in images {
            guard let image = loadImage(from: path) else {
                throw ValidationError("Failed to load image: \(path)")
            }
            refImages.append(image)
            if verbose {
                print("Loaded reference image: \(path) (\(image.width)x\(image.height))")
            }
        }

        // Validate interpret image paths exist (VLM will load them directly)
        var interpretImagePaths: [String] = []
        for path in interpret {
            guard FileManager.default.fileExists(atPath: path) else {
                throw ValidationError("Interpret image not found: \(path)")
            }
            interpretImagePaths.append(path)
            if verbose {
                print("Will interpret image: \(path) [VLM analysis]")
            }
        }

        // Show output dimensions
        let outputWidth = width ?? refImages[0].width
        let outputHeight = height ?? refImages[0].height

        print("I2I \(outputWidth)x\(outputHeight), \(actualSteps) steps, guidance \(actualGuidance), \(refImages.count) ref image(s)\(seed.map { ", seed \($0)" } ?? "")...")
        if verbose {
            print("  Prompt: \"\(prompt)\"")
            if let negativePrompt, !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("  Negative prompt: \"\(negativePrompt)\"")
            }
            if upsamplePrompt {
                print("  Prompt upsampling: enabled")
                print("  Prompt upsampler: \(promptUpsampler.model.displayName)\(promptUpsampler.path.map { " [\($0.path)]" } ?? "")")
            }
        }

        // Parse quantization
        guard let textQuantization = MistralQuantization(rawValue: textQuant) else {
            throw ValidationError("Invalid text quantization: \(textQuant). Use: bf16, 8bit, 6bit, 4bit")
        }

        let transformerQuantization = try TransformerQuantization.parseCLI(transformerQuant)

        let quantConfig = Flux2QuantizationConfig(
            textEncoder: textQuantization,
            transformer: transformerQuantization
        )

        // Warn if bf16 with insufficient RAM
        if transformerQuantization == .bf16 {
            let systemRAM = ModelRegistry.systemRAMGB
            if systemRAM < 96 {
                print("⚠️  Warning: bf16 transformer requires ~90GB RAM, you have \(systemRAM)GB")
                print("   Consider using --transformer-quant qint8 for lower memory usage")
                print()
            }
        }

        // Print LoRA info if specified
        if verbose, !loraConfigs.isEmpty {
            print("LoRAs: \(loraConfigs.count)")
            for config in loraConfigs {
                print("  - \(config.name) (scale: \(config.effectiveScale))")
                if let overrides = config.schedulerOverrides, overrides.hasOverrides {
                    if overrides.customSigmas != nil {
                        print("    - Custom sigmas: \(overrides.customSigmas!.count) values")
                    }
                    if let recSteps = overrides.numSteps, steps == nil {
                        print("    - Using recommended steps: \(recSteps)")
                    }
                    if let recGuidance = overrides.guidance, guidance == nil {
                        print("    - Using recommended guidance: \(recGuidance)")
                    }
                }
            }
        }

        // Parse VAE variant
        guard let vaeVar = ModelRegistry.VAEVariant(rawValue: vaeVariant) else {
            throw ValidationError("Invalid VAE variant: \(vaeVariant). Use standard or small-decoder")
        }

        // Create pipeline with HuggingFace token
        let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig, vaeVariant: vaeVar, hfToken: token)
        pipeline.promptUpsampler = promptUpsampler.model
        pipeline.promptUpsamplerPath = promptUpsampler.path

        // Set memory profile
        switch memoryProfile.lowercased() {
        case "conservative": pipeline.memoryProfile = .conservative
        case "balanced": pipeline.memoryProfile = .balanced
        case "performance": pipeline.memoryProfile = .performance
        case "auto": pipeline.memoryProfile = .auto
        default:
            throw ValidationError("Invalid memory profile: \(memoryProfile). Use auto, conservative, balanced, or performance")
        }

        // Load all LoRAs before the transformer is loaded so they can be fused
        // together in one pass (important for quantized transformers).
        for config in loraConfigs {
            do {
                let info = try pipeline.loadLoRA(config)
                if verbose {
                    print("LoRA loaded: \(info.numLayers) layers, rank \(info.rank), \(String(format: "%.1f", info.memorySizeMB)) MB — \(config.name)")
                }
                if info.targetModel != .unknown && info.targetModel.rawValue != modelVariant.rawValue {
                    print("⚠️  Warning: LoRA \(config.name) was trained for \(info.targetModel.rawValue), but using \(modelVariant.rawValue)")
                }
            } catch {
                throw ValidationError("Failed to load LoRA \(config.name): \(error.localizedDescription)")
            }
        }

        // Setup checkpoint directory if needed
        let checkpointDir: String?
        if let interval = checkpoint, interval > 0 {
            let baseName = (output as NSString).deletingPathExtension
            checkpointDir = "\(baseName)_checkpoints"
            try FileManager.default.createDirectory(
                atPath: checkpointDir!,
                withIntermediateDirectories: true
            )
            if verbose {
                print("Checkpoints will be saved to: \(checkpointDir!)")
            }
        } else {
            checkpointDir = nil
        }

        // Reference-encode budget policy: framework owns the mechanism (a per-image
        // pixel ceiling); the CLI just maps the user-facing megapixel flag to pixels.
        // 1 MP == 1024×1024, so the default resolves to the historical 1024² budget.
        let maxReferencePixels = maxReferenceMegapixels
            .map { max(32 * 32, Int(($0 * 1024 * 1024).rounded())) } ?? (1024 * 1024)

        let image = try await pipeline.generateImageToImage(
            prompt: prompt,
            images: refImages,
            negativePrompt: negativePrompt,
            interpretImagePaths: interpretImagePaths.isEmpty ? nil : interpretImagePaths,
            height: height,
            width: width,
            steps: actualSteps,
            guidance: actualGuidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpoint,
            maxReferencePixels: maxReferencePixels,
            onProgress: { current, total in
                let progress = Float(current) / Float(total) * 100
                print("\rStep \(current)/\(total) [\(String(format: "%.0f", progress))%]", terminator: "")
                fflush(stdout)
            },
            onCheckpoint: { step, checkpointImage in
                if let dir = checkpointDir {
                    let checkpointPath = "\(dir)/step_\(String(format: "%03d", step)).png"
                    do {
                        try saveImage(checkpointImage, to: checkpointPath)
                        print("\n  Checkpoint saved: step_\(String(format: "%03d", step)).png")
                    } catch {
                        print("\n  Failed to save checkpoint at step \(step): \(error.localizedDescription)")
                    }
                }
            }
        )

        print()

        let elapsed = Date().timeIntervalSince(startTime)
        print("Generation completed in \(String(format: "%.1f", elapsed))s")

        // Save image
        try saveImage(image, to: output)
        print("Image saved to \(output)")

        // Show profiler report if requested
        if profile {
            print()
            print(Flux2Profiler.shared.generateReport())
        }
    }
}

// MARK: - Download Command

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download required models from HuggingFace"
    )

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Model to download: dev, klein-4b, klein-9b")
    var model: String = "dev"

    @Option(name: .long, help: "Transformer quantization: \(TransformerQuantization.cliValueList)")
    var transformerQuant: String = "qint8"

    @Flag(name: .long, help: "Download all model variants")
    var all: Bool = false

    @Flag(name: .long, help: "Only download VAE")
    var vaeOnly: Bool = false

    @Option(name: .long, help: "VAE variant to download: small-decoder (default), standard")
    var vaeVariant: String = "small-decoder"

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage)")
    var modelsDir: String?

    func run() async throws {
        // Configure custom models directory
        configureModelsDirectory(modelsDir)

        // Get token from environment if not provided
        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        // Parse VAE variant
        guard let vaeVar = ModelRegistry.VAEVariant(rawValue: vaeVariant) else {
            throw ValidationError("Invalid VAE variant: \(vaeVariant). Use standard or small-decoder")
        }

        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, or klein-9b")
        }

        // Check if model requires authentication (gated models)
        // Gated: dev, klein-9b (both base and distilled)
        // Non-gated: klein-4b
        let isGatedModel = modelVariant == .dev || modelVariant == .klein9B || modelVariant == .klein9BKV

        if token == nil && isGatedModel {
            print("⚠️  No HuggingFace token provided.")
            print("   \(modelVariant.displayName) is a gated model. You need to:")
            print("   1. Accept the license at https://huggingface.co/black-forest-labs/\(modelVariant == .dev ? "FLUX.2-dev" : "FLUX.2-klein-9B")")
            print("   2. Set HF_TOKEN environment variable or use --hf-token")
            print()
        }

        let downloader = Flux2ModelDownloader(hfToken: token)

        if vaeOnly {
            print("Downloading \(vaeVar.displayName)...")
            let component = ModelRegistry.ModelComponent.vae(vaeVar)
            try await downloadComponent(downloader, component)
            return
        }

        if all {
            print("Downloading all model variants for \(modelVariant.displayName)...")
            let variants = ModelRegistry.TransformerVariant.allCases.filter { $0.modelType == modelVariant }
            for variant in variants {
                let component = ModelRegistry.ModelComponent.transformer(variant)
                try await downloadComponent(downloader, component)
            }
        } else {
            // Parse quantization and get the right variant for this model type
            let quant = try TransformerQuantization.parseCLI(transformerQuant)

            let variant = ModelRegistry.TransformerVariant.variant(for: modelVariant, quantization: quant)
            print("Downloading \(modelVariant.displayName) Transformer (\(variant.rawValue))...")
            let component = ModelRegistry.ModelComponent.transformer(variant)
            try await downloadComponent(downloader, component)
        }

        // Always download VAE
        print("Downloading \(vaeVar.displayName)...")
        let vaeComponent = ModelRegistry.ModelComponent.vae(vaeVar)
        try await downloadComponent(downloader, vaeComponent)

        print()
        print("✅ Download complete!")
        print("   Models stored in: \(ModelRegistry.modelsDirectory.path)")

        // Text encoder info
        switch modelVariant {
        case .dev:
            print("   Note: Text encoder (Mistral) will be auto-downloaded on first run")
        case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
            print("   Note: Text encoder (Qwen3) will be auto-downloaded on first run")
        }
    }

    private func downloadComponent(_ downloader: Flux2ModelDownloader, _ component: ModelRegistry.ModelComponent) async throws {
        do {
            _ = try await downloader.download(component) { progress, message in
                let percent = Int(progress * 100)
                print("\r  [\(percent)%] \(message)", terminator: "")
                fflush(stdout)
            }
            print()
        } catch {
            print()
            print("❌ Failed to download \(component.displayName): \(error.localizedDescription)")
            throw error
        }
    }
}

// MARK: - Info Command

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show system and model information"
    )

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage)")
    var modelsDir: String?

    func run() throws {
        // Configure custom models directory
        configureModelsDirectory(modelsDir)
        print("Flux.2 Swift MLX Framework")
        print("Version: \(Flux2Core.version)")
        print()

        print("System Information:")
        print("  RAM: \(ModelRegistry.systemRAMGB)GB")
        print("  Recommended config: \(ModelRegistry.defaultConfig.description)")
        print()

        print("Available Quantization Presets:")
        print("  High Quality (~90GB): bf16 text + bf16 transformer")
        print("  Balanced (~57GB): 8bit text + qint8 transformer")
        print("  Memory Efficient (~47GB): 4bit text + qint8 transformer")
        print("  Minimal (~47GB): 4bit text + qint8 transformer")
        print("  Ultra-Minimal (~30GB): 4bit text + int4 transformer")
        print()

        print("Model Status:")
        for variant in ModelRegistry.TransformerVariant.allCases {
            let component = ModelRegistry.ModelComponent.transformer(variant)
            let status = ModelRegistry.isDownloaded(component) ? "✓" : "✗"
            print("  [\(status)] \(component.displayName)")
        }

        for variant in ModelRegistry.TextEncoderVariant.allCases {
            let component = ModelRegistry.ModelComponent.textEncoder(variant)
            let status = ModelRegistry.isDownloaded(component) ? "✓" : "✗"
            print("  [\(status)] \(component.displayName)")
        }

        for variant in ModelRegistry.VAEVariant.allCases {
            let component = ModelRegistry.ModelComponent.vae(variant)
            let status = ModelRegistry.isDownloaded(component) ? "✓" : "✗"
            print("  [\(status)] \(component.displayName)")
        }
    }
}

// MARK: - Helper Functions

func loadImage(from path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }
    return image
}

func saveImage(_ image: CGImage, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    let utType: CFString = path.hasSuffix(".png") ? UTType.png.identifier as CFString : UTType.jpeg.identifier as CFString
    let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        utType,
        1,
        nil
    )

    guard let dest = destination else {
        throw Flux2Error.imageProcessingFailed("Failed to create image destination")
    }

    CGImageDestinationAddImage(dest, image, nil)

    guard CGImageDestinationFinalize(dest) else {
        throw Flux2Error.imageProcessingFailed("Failed to write image")
    }
}
