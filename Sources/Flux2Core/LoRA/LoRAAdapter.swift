// LoRAAdapter.swift - Apply LoRA weights to transformer layers
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Applies LoRA weights to a base linear layer
///
/// LoRA (Low-Rank Adaptation) adds low-rank updates to model weights:
/// `output = base_linear(x) + scale * lora_B(lora_A(x))`
///
/// This allows efficient fine-tuning without modifying the base model weights.
public class LoRALinear: Module, @unchecked Sendable {
    /// The base linear layer (frozen)
    let base: Linear

    /// LoRA A matrix (down projection): [rank, input_dim]
    let loraA: MLXArray

    /// LoRA B matrix (up projection): [output_dim, rank]
    let loraB: MLXArray

    /// Scale factor for LoRA output
    var scale: Float

    /// Initialize LoRA-enhanced linear layer
    /// - Parameters:
    ///   - base: The original linear layer
    ///   - loraA: LoRA A matrix
    ///   - loraB: LoRA B matrix
    ///   - scale: Scale factor for LoRA output
    public init(base: Linear, loraA: MLXArray, loraB: MLXArray, scale: Float = 1.0) {
        self.base = base
        self.loraA = loraA
        self.loraB = loraB
        self.scale = scale
    }

    /// Forward pass with LoRA adaptation
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear output
        let baseOutput = base(x)

        // LoRA adaptation: scale * (x @ loraA.T @ loraB.T)
        // loraA: [rank, in_features] -> transpose to [in_features, rank]
        // loraB: [out_features, rank] -> transpose to [rank, out_features]
        let loraOutput = matmul(matmul(x, loraA.T), loraB.T)

        return baseOutput + scale * loraOutput
    }
}

/// A fused update supplied by a LoRA adapter.
///
/// Standard LoRA stores a low-rank B @ A pair.  LyCORIS/ai-toolkit LoKr
/// stores two Kronecker factors instead; expanding those factors is deferred
/// until the target model weight is merged so we never retain a full-size
/// (and often multi-hundred-MB) delta just to load the adapter.
enum LoRAWeightUpdate {
    case lowRank(scale: Float, loraA: MLXArray, loraB: MLXArray)
    case kronecker(scale: Float, w1: MLXArray, w2: MLXArray)
}

/// Manages multiple LoRA adapters and their application to a transformer
public class LoRAManager: @unchecked Sendable {
    /// Loaded LoRA loaders
    private var loaders: [LoRALoader] = []

    /// Layer path to standard LoRA mappings (for lookup during forward)
    private var layerMappings: [String: [(loader: LoRALoader, pair: LoRAWeightPair)]] = [:]

    /// Layer path to LoKr mappings.  Kept separate because LoKr has no
    /// lora_A/lora_B pair and must be expanded with MLX.kron at fusion time.
    private var lokrLayerMappings: [String: [(loader: LoRALoader, pair: LoKRWeightPair)]] = [:]

    /// Initialize empty LoRA manager
    public init() {}

    /// Load and register a LoRA
    /// - Parameter config: LoRA configuration
    /// - Returns: LoRA info after loading
    @discardableResult
    public func loadLoRA(_ config: LoRAConfig) throws -> LoRAInfo {
        let loader = LoRALoader(config: config)
        try loader.load()

        guard let info = loader.info else {
            throw LoRALoaderError.invalidFormat("Failed to get LoRA info")
        }

        loaders.append(loader)

        // Register standard LoRA layer mappings.
        for path in loader.layerPaths {
            if layerMappings[path] == nil {
                layerMappings[path] = []
            }
            if let pair = loader.getWeights(for: path) {
                layerMappings[path]?.append((loader: loader, pair: pair))
            }
        }

        // Register direct LoKr mappings separately.  A LoKr update is not a
        // low-rank B @ A pair, so it must remain as factors until fusion.
        for path in loader.lokrLayerPaths {
            if lokrLayerMappings[path] == nil {
                lokrLayerMappings[path] = []
            }
            if let pair = loader.getLoKRWeights(for: path) {
                lokrLayerMappings[path]?.append((loader: loader, pair: pair))
            }
        }

        Flux2Debug.log("[LoRA] Registered \(config.name) with \(info.numLayers) layers")

        return info
    }

    /// Unload a LoRA by name
    public func unloadLoRA(name: String) {
        loaders.removeAll { $0.config.name == name }
        rebuildMappings()
    }

    /// Unload all LoRAs
    public func unloadAll() {
        loaders.removeAll()
        layerMappings.removeAll()
        lokrLayerMappings.removeAll()
    }

    /// Rebuild layer mappings after unload
    private func rebuildMappings() {
        layerMappings.removeAll()
        lokrLayerMappings.removeAll()
        for loader in loaders {
            for path in loader.layerPaths {
                if layerMappings[path] == nil {
                    layerMappings[path] = []
                }
                if let pair = loader.getWeights(for: path) {
                    layerMappings[path]?.append((loader: loader, pair: pair))
                }
            }
            for path in loader.lokrLayerPaths {
                if lokrLayerMappings[path] == nil {
                    lokrLayerMappings[path] = []
                }
                if let pair = loader.getLoKRWeights(for: path) {
                    lokrLayerMappings[path]?.append((loader: loader, pair: pair))
                }
            }
        }
    }

    /// Update scale for a specific LoRA
    public func setScale(name: String, scale: Float) {
        for loader in loaders {
            if loader.config.name == name {
                // LoRAConfig is a struct, need to work around mutability
                // For now, we'll track scales separately
                // TODO: Make scale mutable in LoRAConfig
            }
        }
    }

    /// Check if a layer has LoRA adapters
    public func hasLoRA(for layerPath: String) -> Bool {
        let hasStandard = layerMappings[layerPath]?.isEmpty == false
        let hasLoKR = lokrLayerMappings[layerPath]?.isEmpty == false
        return hasStandard || hasLoKR
    }

    /// Get all standard LoRA pairs for a layer path.
    public func getLoRAPairs(for layerPath: String) -> [(scale: Float, loraA: MLXArray, loraB: MLXArray)] {
        guard let mappings = layerMappings[layerPath] else { return [] }

        return mappings.map { mapping in
            // Effective scale = user scale * metadata scale (alpha/rank)
            // This ensures LoRAs trained with different alpha/rank values work correctly
            let effectiveScale = mapping.loader.config.effectiveScale * mapping.loader.metadataScale
            return (
                scale: effectiveScale,
                loraA: mapping.pair.loraA.array,
                loraB: mapping.pair.loraB.array
            )
        }
    }

    /// Get all update forms for a layer path.  This is consumed by the fusion
    /// path, which also understands direct Kronecker factors.
    func getLoRAUpdates(for layerPath: String) -> [LoRAWeightUpdate] {
        var updates: [LoRAWeightUpdate] = []

        if let mappings = layerMappings[layerPath] {
            updates.append(contentsOf: mappings.map { mapping in
                let scale = mapping.loader.config.effectiveScale * mapping.loader.metadataScale
                return .lowRank(
                    scale: scale,
                    loraA: mapping.pair.loraA.array,
                    loraB: mapping.pair.loraB.array)
            })
        }

        if let mappings = lokrLayerMappings[layerPath] {
            updates.append(contentsOf: mappings.map { mapping in
                // Direct LoKr exports (w1 + w2) intentionally ignore the
                // per-layer alpha tensor.  This matches LyCORIS/ComfyUI:
                // alpha is only a normalization factor for decomposed LoKr.
                return .kronecker(
                    scale: mapping.loader.config.effectiveScale,
                    w1: mapping.pair.w1.array,
                    w2: mapping.pair.w2.array)
            })
        }

        return updates
    }

    /// Apply LoRA to a linear layer output
    /// - Parameters:
    ///   - baseOutput: Output from the base linear layer
    ///   - input: Original input to the linear layer
    ///   - layerPath: Path identifying the layer
    /// - Returns: Output with LoRA adaptation applied
    public func applyLoRA(baseOutput: MLXArray, input: MLXArray, layerPath: String) -> MLXArray {
        let updates = getLoRAUpdates(for: layerPath)
        guard !updates.isEmpty else { return baseOutput }

        var output = baseOutput
        for update in updates {
            switch update {
            case .lowRank(let scale, let loraA, let loraB):
                // LoRA: output += scale * (input @ loraA.T @ loraB.T)
                let loraOutput = matmul(matmul(input, loraA.T), loraB.T)
                output = output + scale * loraOutput

            case .kronecker(let scale, let w1, let w2):
                // Apply kron(w1, w2) without materializing the full matrix.
                // For x[..., in_m * in_n], kron(w1, w2) has shape
                // [out_l * out_k, in_m * in_n].
                guard w1.ndim == 2, w2.ndim == 2 else { continue }
                let inputShape = Array(input.shape.dropLast()) + [w1.shape[1], w2.shape[1]]
                let grouped = input.reshaped(inputShape)
                let afterW2 = matmul(grouped, w2.T)
                let afterW1 = matmul(afterW2.swappedAxes(-1, -2), w1.T)
                let delta = afterW1.swappedAxes(-1, -2)
                    .reshaped(Array(input.shape.dropLast()) + [w1.shape[0] * w2.shape[0]])
                output = output + scale * delta
            }
        }

        return output
    }

    /// Number of loaded LoRAs
    public var count: Int {
        loaders.count
    }

    /// Names of loaded LoRAs
    public var loadedNames: [String] {
        loaders.map { $0.config.name }
    }

    /// All layer paths that have LoRA weights
    public var loadedLayerPaths: [String] {
        Array(Set(layerMappings.keys).union(lokrLayerMappings.keys)).sorted()
    }

    /// Get combined activation keywords from all loaded LoRAs
    public var activationKeywords: [String] {
        loaders.compactMap { $0.config.activationKeyword }
    }

    /// Prepend activation keywords to a prompt
    public func enhancePrompt(_ prompt: String) -> String {
        let keywords = activationKeywords
        if keywords.isEmpty { return prompt }
        return keywords.joined(separator: ", ") + ", " + prompt
    }

    /// Clear LoRA weights from memory after fusion into base model
    ///
    /// After LoRA weights have been merged into the transformer via `mergeLoRAWeights`,
    /// the original LoRA matrices are no longer needed. This method frees that memory.
    ///
    /// - Note: After calling this, LoRAs cannot be "unfused" without reloading the base model.
    public func clearWeightsAfterFusion() {
        var totalFreed: Float = 0
        for loader in loaders {
            if loader.hasWeightsInMemory {
                totalFreed += loader.info?.memorySizeMB ?? 0
                loader.clearWeightsAfterFusion()
            }
        }
        layerMappings.removeAll()
        lokrLayerMappings.removeAll()
        Flux2Debug.log("[LoRA] Total memory freed after fusion: ~\(String(format: "%.1f", totalFreed)) MB")
    }

    /// Whether any loaders still have weights in memory
    public var hasWeightsInMemory: Bool {
        loaders.contains { $0.hasWeightsInMemory }
    }
}
