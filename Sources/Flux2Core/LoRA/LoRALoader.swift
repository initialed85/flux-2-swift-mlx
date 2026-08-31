// LoRALoader.swift - Load LoRA weights from safetensors
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Errors that can occur during LoRA loading
public enum LoRALoaderError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)
    case incompatibleModel(String)
    case missingWeights(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "LoRA file not found: \(path)"
        case .invalidFormat(let message):
            return "Invalid LoRA format: \(message)"
        case .incompatibleModel(let message):
            return "Incompatible model: \(message)"
        case .missingWeights(let layer):
            return "Missing LoRA weights for layer: \(layer)"
        }
    }
}

/// Loads LoRA weights from safetensors files
public class LoRALoader {

    /// Loaded standard LoRA weights, keyed by target layer path
    private(set) var weights: [String: LoRAWeightPair] = [:]

    /// Loaded direct LoKr factors, keyed by target layer path
    private(set) var lokrWeights: [String: LoKRWeightPair] = [:]

    /// LoRA information
    private(set) var info: LoRAInfo?

    /// The LoRA configuration
    public let config: LoRAConfig
    
    /// Scale factor derived from metadata (alpha/rank), defaults to 1.0
    /// This is computed from the safetensors metadata if available
    private(set) var metadataScale: Float = 1.0

    /// Initialize LoRA loader with configuration
    public init(config: LoRAConfig) {
        self.config = config
    }

    /// Load LoRA weights from file
    public func load() throws {
        let path = config.filePath

        guard FileManager.default.fileExists(atPath: path) else {
            throw LoRALoaderError.fileNotFound(path)
        }

        Flux2Debug.log("[LoRA] Loading from: \(path)")

        // Load safetensors with metadata
        let (rawWeights, metadata) = try loadArraysAndMetadata(url: URL(fileURLWithPath: path))
        
        // Parse metadata for alpha/rank if available
        parseMetadata(metadata)

        Flux2Debug.log("[LoRA] Loaded \(rawWeights.count) tensors")

        // Parse LoRA weights and map to layer paths
        try parseAndMapWeights(rawWeights)

        let mappedCount = weights.count + lokrWeights.count
        guard mappedCount > 0 else {
            throw LoRALoaderError.invalidFormat(
                "No supported LoRA layers found. Expected lora_A/lora_B or direct lokr_w1/lokr_w2 tensors.")
        }
        Flux2Debug.log("[LoRA] Mapped \(mappedCount) layer pairs (LoRA: \(weights.count), LoKr: \(lokrWeights.count))")
    }
    
    /// Parse metadata to extract LoRA scale factor (alpha/rank)
    private func parseMetadata(_ metadata: [String: String]) {
        // Try to extract alpha and rank from metadata
        if let alphaStr = metadata["lora_alpha"], let rankStr = metadata["lora_rank"],
           let alpha = Float(alphaStr), let rank = Float(rankStr), rank > 0 {
            metadataScale = alpha / rank
            Flux2Debug.log("[LoRA] Found metadata: alpha=\(alpha), rank=\(rank), computed scale=\(metadataScale)")
        } else {
            // Default to 1.0 if no metadata
            metadataScale = 1.0
            Flux2Debug.verbose("[LoRA] No alpha/rank metadata found, using default scale=1.0")
        }
        
        // Log other useful metadata
        if let format = metadata["format"] {
            Flux2Debug.verbose("[LoRA] Format: \(format)")
        }
        if let software = metadata["software"] {
            Flux2Debug.verbose("[LoRA] Created with: \(software)")
        }
    }

    /// Detect if LoRA uses Diffusers format (vs BFL format)
    private func isDiffusersFormat(_ keys: [String]) -> Bool {
        // Diffusers format uses "transformer.transformer_blocks" or "transformer.single_transformer_blocks"
        // BFL format uses "double_blocks" or "single_blocks" directly
        return keys.contains { $0.contains("transformer_blocks") || $0.contains("single_transformer_blocks") }
    }

    /// Parse raw weights and map to our layer naming scheme.
    ///
    /// In addition to ordinary PEFT LoRA (lora_A/lora_B), this accepts the
    /// direct LoKr format emitted by ai-toolkit/LyCORIS:
    /// `*.lokr_w1`, `*.lokr_w2`, and a per-layer `*.alpha`.  The alpha tensor
    /// is intentionally not retained for direct factors; LyCORIS uses alpha
    /// only to normalize decomposed factors, while direct w1/w2 exports are
    /// already at their runtime scale.
    private func parseAndMapWeights(_ rawWeights: [String: MLXArray]) throws {
        var layerGroups: [String: (loraA: MLXArray?, loraB: MLXArray?)] = [:]
        var lokrGroups: [String: (w1: MLXArray?, w2: MLXArray?)] = [:]
        var formatKeys: [String] = []

        for (key, value) in rawWeights {
            // Skip metadata keys.
            if key.hasPrefix("__") { continue }

            let basePath: String
            let kind: String
            if let standard = splitStandardLoRAKey(key) {
                // PEFT/diffusers commonly inserts an adapter name between the
                // matrix kind and `.weight`, e.g. `lora_A.default.weight`.
                basePath = standard.basePath
                kind = standard.kind
            } else if key.hasSuffix(".lokr_w1") {
                basePath = String(key.dropLast(".lokr_w1".count))
                kind = "lokrW1"
            } else if key.hasSuffix(".lokr_w2") {
                basePath = String(key.dropLast(".lokr_w2".count))
                kind = "lokrW2"
            } else if key.hasSuffix(".alpha") {
                // Direct LoKr alpha is deliberately ignored.  Still record
                // the path for format detection and diagnostics.
                basePath = String(key.dropLast(".alpha".count))
                kind = "alpha"
            } else {
                Flux2Debug.verbose("[LoRA] Skipping non-LoRA key: \(key)")
                continue
            }

            let layerPath = normalizeLayerPath(basePath)
            formatKeys.append(layerPath)

            switch kind {
            case "loraA":
                layerGroups[layerPath, default: (nil, nil)].loraA = value
            case "loraB":
                layerGroups[layerPath, default: (nil, nil)].loraB = value
            case "lokrW1":
                lokrGroups[layerPath, default: (nil, nil)].w1 = value
            case "lokrW2":
                lokrGroups[layerPath, default: (nil, nil)].w2 = value
            default:
                break
            }
        }

        // Detect format from normalized layer paths (after common prefixes
        // have been removed, BFL uses double_blocks/single_blocks).
        let isDiffusers = isDiffusersFormat(formatKeys)
        Flux2Debug.log("[LoRA] Detected format: \(isDiffusers ? "Diffusers" : "BFL")")

        var rank: Int = 0
        var totalParams = 0

        // Convert standard LoRA pairs.
        for (rawPath, pair) in layerGroups {
            guard let loraA = pair.loraA, let loraB = pair.loraB else {
                Flux2Debug.log("[LoRA] Warning: Missing pair for \(rawPath)")
                continue
            }

            if rank == 0 {
                rank = loraA.shape[0]
                Flux2Debug.log("[LoRA] Detected rank: \(rank)")
                Flux2Debug.log("[LoRA] loraA dtype: \(loraA.dtype), loraB dtype: \(loraB.dtype)")
            }

            if isDiffusers {
                let swiftPath = mapDiffusersPathToSwiftPath(rawPath)
                weights[swiftPath] = LoRAWeightPair(
                    loraA: MLXArrayWrapper(loraA), loraB: MLXArrayWrapper(loraB))
                totalParams += loraA.size + loraB.size
            } else if isCombinedQKVLayer(rawPath) {
                let splitPairs = splitQKVLoRA(bflPath: rawPath, loraA: loraA, loraB: loraB)
                for (swiftPath, splitLoraA, splitLoraB) in splitPairs {
                    weights[swiftPath] = LoRAWeightPair(
                        loraA: MLXArrayWrapper(splitLoraA), loraB: MLXArrayWrapper(splitLoraB))
                    totalParams += splitLoraA.size + splitLoraB.size
                }
            } else {
                let swiftPath = mapBFLPathToSwiftPath(rawPath)
                weights[swiftPath] = LoRAWeightPair(
                    loraA: MLXArrayWrapper(loraA), loraB: MLXArrayWrapper(loraB))
                totalParams += loraA.size + loraB.size
            }
        }

        // Convert direct LoKr pairs.  Combined BFL QKV paths are kept as one
        // pair and split after kron expansion during fusion; splitting w2
        // before expansion would produce the wrong row ordering.
        for (rawPath, pair) in lokrGroups {
            guard let w1 = pair.w1, let w2 = pair.w2 else {
                Flux2Debug.log("[LoRA] Warning: Missing LoKr factor pair for \(rawPath)")
                continue
            }

            if rank == 0 {
                rank = w1.shape.first ?? 0
                Flux2Debug.log("[LoRA] Detected LoKr rank/factor: \(rank)")
                Flux2Debug.log("[LoRA] w1 dtype: \(w1.dtype), w2 dtype: \(w2.dtype)")
            }

            let swiftPath: String
            if isDiffusers {
                swiftPath = mapDiffusersPathToSwiftPath(rawPath)
            } else if isCombinedQKVLayer(rawPath) {
                // Keep fused BFL QKV paths intact.  They are split only after
                // kron expansion because w2's rows are interleaved by w1.
                swiftPath = rawPath
            } else {
                swiftPath = mapBFLPathToSwiftPath(rawPath)
            }
            lokrWeights[swiftPath] = LoKRWeightPair(
                w1: MLXArrayWrapper(w1), w2: MLXArrayWrapper(w2))
            totalParams += w1.size + w2.size
        }

        let targetModel = detectTargetModel(formatKeys)
        let mappedLayers = weights.count + lokrWeights.count
        info = LoRAInfo(
            numLayers: mappedLayers,
            rank: rank,
            numParameters: totalParams,
            targetModel: targetModel)

        Flux2Debug.log("[LoRA] Info: \(mappedLayers) layers, rank=\(rank), params=\(totalParams), target=\(targetModel.rawValue)")
    }

    /// Parse conventional PEFT keys with either `lora_A.weight` or an
    /// adapter-qualified `lora_A.<adapter>.weight` suffix.
    private func splitStandardLoRAKey(_ key: String) -> (basePath: String, kind: String)? {
        let components = key.split(separator: ".").map(String.init)
        guard components.last == "weight", components.count >= 3 else {
            return nil
        }

        var markerIndex: Int?
        for index in components.indices.reversed() {
            if components[index] == "lora_A" || components[index] == "lora_B" {
                markerIndex = index
                break
            }
        }
        guard let markerIndex, markerIndex > 0, markerIndex < components.count - 1 else {
            return nil
        }

        let basePath = components[..<markerIndex].joined(separator: ".")
        let kind = components[markerIndex] == "lora_A" ? "loraA" : "loraB"
        return (basePath, kind)
    }

    /// Strip prefixes used by PEFT/diffusers exports while retaining the
    /// architecture-specific BFL or diffusers layer names for mapping.
    private func normalizeLayerPath(_ path: String) -> String {
        var layerPath = path
        if layerPath.hasPrefix("base_model.model.") {
            layerPath = String(layerPath.dropFirst("base_model.model.".count))
        }
        if layerPath.hasPrefix("transformer.") {
            layerPath = String(layerPath.dropFirst("transformer.".count))
        }
        if layerPath.hasPrefix("diffusion_model.") {
            layerPath = String(layerPath.dropFirst("diffusion_model.".count))
        }
        return layerPath
    }

    /// Check if a layer path refers to a combined QKV projection
    private func isCombinedQKVLayer(_ bflPath: String) -> Bool {
        return bflPath.contains(".img_attn.qkv") || bflPath.contains(".txt_attn.qkv")
    }

    /// Split combined QKV LoRA weights into separate Q, K, V parts
    /// The loraB output dimension is 3x the individual projection dimension
    private func splitQKVLoRA(bflPath: String, loraA: MLXArray, loraB: MLXArray) -> [(String, MLXArray, MLXArray)] {
        let blockIdx = extractBlockIndex(from: bflPath, prefix: "double_blocks.")

        // loraB shape: [3*innerDim, rank] - split along first axis
        let totalOutputDim = loraB.shape[0]
        let singleDim = totalOutputDim / 3

        // Split loraB into Q, K, V parts
        let loraBQ = loraB[0..<singleDim, 0...]
        let loraBK = loraB[singleDim..<(singleDim * 2), 0...]
        let loraBV = loraB[(singleDim * 2)..., 0...]

        // loraA is shared for Q, K, V
        if bflPath.contains(".img_attn.qkv") {
            return [
                ("transformerBlocks.\(blockIdx).attn.toQ", loraA, loraBQ),
                ("transformerBlocks.\(blockIdx).attn.toK", loraA, loraBK),
                ("transformerBlocks.\(blockIdx).attn.toV", loraA, loraBV)
            ]
        } else {
            // txt_attn.qkv
            return [
                ("transformerBlocks.\(blockIdx).attn.addQProj", loraA, loraBQ),
                ("transformerBlocks.\(blockIdx).attn.addKProj", loraA, loraBK),
                ("transformerBlocks.\(blockIdx).attn.addVProj", loraA, loraBV)
            ]
        }
    }

    /// Map Diffusers format layer path to Swift module path
    /// Diffusers uses names like:
    ///   - single_transformer_blocks.X.attn.to_qkv_mlp_proj → singleTransformerBlocks.X.attn.toQkvMlp
    ///   - transformer_blocks.X.attn.to_q → transformerBlocks.X.attn.toQ
    ///   - transformer_blocks.X.attn.add_q_proj → transformerBlocks.X.attn.addQProj
    private func mapDiffusersPathToSwiftPath(_ diffusersPath: String) -> String {
        var path = diffusersPath

        // Map block names
        path = path.replacingOccurrences(of: "single_transformer_blocks.", with: "singleTransformerBlocks.")
        path = path.replacingOccurrences(of: "transformer_blocks.", with: "transformerBlocks.")

        // Map attention projections (single blocks)
        path = path.replacingOccurrences(of: ".attn.to_qkv_mlp_proj", with: ".attn.toQkvMlp")
        path = path.replacingOccurrences(of: ".attn.to_qkv_mlp", with: ".attn.toQkvMlp")
        // Note: Single blocks use ".attn.to_out" without .0
        // Double blocks use ".attn.to_out.0" with .0
        path = path.replacingOccurrences(of: ".attn.to_out.0", with: ".attn.toOut")  // Double blocks first (more specific)
        path = path.replacingOccurrences(of: ".attn.to_out", with: ".attn.toOut")    // Single blocks (fallback)

        // Map attention projections (double blocks - image)
        path = path.replacingOccurrences(of: ".attn.to_q", with: ".attn.toQ")
        path = path.replacingOccurrences(of: ".attn.to_k", with: ".attn.toK")
        path = path.replacingOccurrences(of: ".attn.to_v", with: ".attn.toV")
        // Note: ".attn.to_out.0" becomes ".attn.toOut" (the .0 is already stripped)

        // Map attention projections (double blocks - text/context)
        path = path.replacingOccurrences(of: ".attn.add_q_proj", with: ".attn.addQProj")
        path = path.replacingOccurrences(of: ".attn.add_k_proj", with: ".attn.addKProj")
        path = path.replacingOccurrences(of: ".attn.add_v_proj", with: ".attn.addVProj")
        path = path.replacingOccurrences(of: ".attn.to_add_out", with: ".attn.toAddOut")

        // Map FFN layers (trained LoRA uses snake_case, Swift uses camelCase)
        // Note: Order matters - map linear_out before ff_context to handle both
        path = path.replacingOccurrences(of: ".linear_out", with: ".linearOut")
        path = path.replacingOccurrences(of: ".ff_context.", with: ".ffContext.")

        // Map time/guidance embeddings
        path = path.replacingOccurrences(of: "time_guidance_embed.", with: "timeGuidanceEmbed.")
        path = path.replacingOccurrences(of: "time_text_embed.", with: "timeGuidanceEmbed.")
        path = path.replacingOccurrences(of: "timestep_embedder.", with: "timestepEmbedder.")
        path = path.replacingOccurrences(of: "guidance_embedder.", with: "guidanceEmbedder.")
        path = path.replacingOccurrences(of: ".linear_1", with: ".linear1")
        path = path.replacingOccurrences(of: ".linear_2", with: ".linear2")

        // Map modulation layers
        path = path.replacingOccurrences(of: "double_stream_modulation_img.", with: "doubleStreamModulationImg.")
        path = path.replacingOccurrences(of: "double_stream_modulation_txt.", with: "doubleStreamModulationTxt.")
        path = path.replacingOccurrences(of: "single_stream_modulation.", with: "singleStreamModulation.")

        // Map embedders (Diffusers format)
        path = path.replacingOccurrences(of: "x_embedder", with: "xEmbedder")
        path = path.replacingOccurrences(of: "context_embedder", with: "contextEmbedder")

        if path == "img_in" {
            return "xEmbedder"
        }
        if path == "txt_in" {
            return "contextEmbedder"
        }

        // Map final layer (BFL format)
        if path == "final_layer.linear" {
            return "projOut"
        }

        // Map time embeddings (BFL format)
        if path == "time_in.in_layer" {
            return "timeGuidanceEmbed.timestepEmbedder.linear1"
        }
        if path == "time_in.out_layer" {
            return "timeGuidanceEmbed.timestepEmbedder.linear2"
        }

        // Map modulation .lin to .linear (only at the end of path)
        // BFL uses .lin, Swift uses .linear for modulation layers
        if path.hasSuffix(".lin") {
            path = String(path.dropLast(4)) + ".linear"
        }

        // Map output (Diffusers format)
        path = path.replacingOccurrences(of: "norm_out.", with: "normOut.")
        path = path.replacingOccurrences(of: "proj_out", with: "projOut")

        return path
    }

    /// Map BFL layer path to Swift module path
    /// Note: Combined QKV layers (.img_attn.qkv, .txt_attn.qkv) are handled
    /// separately by splitQKVLoRA() and should not reach this function.
    private func mapBFLPathToSwiftPath(_ bflPath: String) -> String {
        // Double block layers (attention + MLP)
        if bflPath.contains("double_blocks.") {
            let blockIdx = extractBlockIndex(from: bflPath, prefix: "double_blocks.")

            // Attention output projections
            if bflPath.contains(".img_attn.proj") {
                return "transformerBlocks.\(blockIdx).attn.toOut"
            } else if bflPath.contains(".txt_attn.proj") {
                return "transformerBlocks.\(blockIdx).attn.toAddOut"
            }
            // Note: .img_attn.qkv and .txt_attn.qkv are split into Q/K/V in splitQKVLoRA()

            // MLP/FFN layers (image stream)
            // BFL: img_mlp.0 is the gated linear (SwiGLU proj), img_mlp.2 is the output linear
            if bflPath.contains(".img_mlp.0") {
                return "transformerBlocks.\(blockIdx).ff.activation.proj"
            } else if bflPath.contains(".img_mlp.2") {
                return "transformerBlocks.\(blockIdx).ff.linearOut"
            }

            // MLP/FFN layers (text/context stream)
            if bflPath.contains(".txt_mlp.0") {
                return "transformerBlocks.\(blockIdx).ffContext.activation.proj"
            } else if bflPath.contains(".txt_mlp.2") {
                return "transformerBlocks.\(blockIdx).ffContext.linearOut"
            }
        }

        // Single block linear layers
        if bflPath.contains("single_blocks.") {
            let blockIdx = extractBlockIndex(from: bflPath, prefix: "single_blocks.")

            if bflPath.contains(".linear1") {
                return "singleTransformerBlocks.\(blockIdx).attn.toQkvMlp"
            } else if bflPath.contains(".linear2") {
                return "singleTransformerBlocks.\(blockIdx).attn.toOut"
            }
        }

        // Modulation layers
        if bflPath.contains("double_stream_modulation_img") {
            return "doubleStreamModulationImg.linear"
        } else if bflPath.contains("double_stream_modulation_txt") {
            return "doubleStreamModulationTxt.linear"
        } else if bflPath.contains("single_stream_modulation") {
            return "singleStreamModulation.linear"
        }

        // Input/output layers
        if bflPath == "img_in" {
            return "xEmbedder"
        } else if bflPath == "txt_in" {
            return "contextEmbedder"
        } else if bflPath == "final_layer.linear" {
            return "projOut"
        }

        // Time embeddings
        if bflPath.contains("time_in.in_layer") {
            return "timeGuidanceEmbed.timestepEmbedder.inLayer"
        } else if bflPath.contains("time_in.out_layer") {
            return "timeGuidanceEmbed.timestepEmbedder.outLayer"
        }

        // Return original path if no mapping found
        Flux2Debug.verbose("[LoRA] No mapping for: \(bflPath)")
        return bflPath
    }

    /// Extract block index from layer path
    private func extractBlockIndex(from path: String, prefix: String) -> Int {
        guard let startRange = path.range(of: prefix) else { return 0 }
        let afterPrefix = path[startRange.upperBound...]
        let indexStr = afterPrefix.prefix(while: { $0.isNumber })
        return Int(indexStr) ?? 0
    }

    /// Detect target model based on layer structure
    private func detectTargetModel(_ layers: [String]) -> LoRAInfo.TargetModel {
        // Count double and single blocks (support both BFL and Diffusers formats)
        var doubleBlocks = Set<Int>()
        var singleBlocks = Set<Int>()

        for path in layers {
            // BFL format: double_blocks.X, single_blocks.X
            if path.hasPrefix("double_blocks.") {
                doubleBlocks.insert(extractBlockIndex(from: path, prefix: "double_blocks."))
            } else if path.hasPrefix("single_blocks.") {
                singleBlocks.insert(extractBlockIndex(from: path, prefix: "single_blocks."))
            }
            // Diffusers format: transformer_blocks.X, single_transformer_blocks.X
            else if path.hasPrefix("transformer_blocks.") {
                doubleBlocks.insert(extractBlockIndex(from: path, prefix: "transformer_blocks."))
            } else if path.hasPrefix("single_transformer_blocks.") {
                singleBlocks.insert(extractBlockIndex(from: path, prefix: "single_transformer_blocks."))
            }
        }

        let maxDouble = doubleBlocks.max() ?? 0
        let maxSingle = singleBlocks.max() ?? 0

        Flux2Debug.log("[LoRA] Detected structure: \(maxDouble + 1) double blocks, \(maxSingle + 1) single blocks")

        // Klein 4B: 5 double, 20 single
        // Klein 9B: 8 double, 24 single
        // Dev: 8 double, 48 single

        if maxDouble == 4 && maxSingle == 19 {
            return .klein4B
        } else if maxDouble == 7 && maxSingle == 23 {
            return .klein9B
        } else if maxDouble == 7 && maxSingle == 47 {
            return .dev
        }

        return .unknown
    }

    /// Get LoRA weight pair for a specific layer
    public func getWeights(for layerPath: String) -> LoRAWeightPair? {
        return weights[layerPath]
    }

    /// Get all layer paths that have standard LoRA weights
    public var layerPaths: [String] {
        Array(weights.keys).sorted()
    }

    /// Get all layer paths that have direct LoKr factors
    var lokrLayerPaths: [String] {
        Array(lokrWeights.keys).sorted()
    }

    /// Get direct LoKr factors for a specific layer
    func getLoKRWeights(for layerPath: String) -> LoKRWeightPair? {
        lokrWeights[layerPath]
    }

    /// Clear LoRA weights from memory after fusion
    ///
    /// After LoRA weights have been merged into the base model weights via `mergeLoRAWeights`,
    /// the original LoRA matrices (loraA/loraB) are no longer needed for inference.
    /// Calling this method frees the memory used by these matrices.
    ///
    /// - Note: After calling this, the LoRA cannot be "unfused" without reloading the base model.
    public func clearWeightsAfterFusion() {
        let memoryFreed = info?.memorySizeMB ?? 0
        weights.removeAll()
        lokrWeights.removeAll()
        Flux2Debug.log("[LoRA] Cleared weights after fusion, freed ~\(String(format: "%.1f", memoryFreed)) MB")
    }

    /// Whether weights are still in memory (not yet cleared after fusion)
    public var hasWeightsInMemory: Bool {
        !weights.isEmpty || !lokrWeights.isEmpty
    }
}
