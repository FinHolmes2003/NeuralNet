#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <sstream>

namespace Network
{
    using json = nlohmann::json;
    using std::vector;
    using std::pair;
    using std::string;
    using std::function;
    using std::invalid_argument;

    float Softsign(const float inputValue) // Essentially sigmoid but faster
    {
        return inputValue / (1.0f + abs(inputValue));
    }

    float InverseSoftsign(const float inputValue)
    {
        float denom = 1.0f + abs(inputValue);
        return 1.0f / (denom * denom);
    }

    /// @brief Converts the json file into a pair of each layers's weights and biases
    /// @param weightsFile json object of the weights and biases file
    /// @return Layer's weights and biases
    pair<vector<vector<float>>, vector<vector<float>>> JsonToVectors(const json& weightsFile)
    {
        size_t numLayers = weightsFile["Layers"].size();

        vector<vector<float>> biases(numLayers);
        vector<vector<float>> weights(numLayers);

        for (size_t i = 0; i < numLayers; i++)
        {
            biases[i] = weightsFile["Layers"][i]["Biases"].get<vector<float>>();
            
            auto weights2D = weightsFile["Layers"][i]["Weights"].get<vector<vector<float>>>();
            size_t numInputs = weights2D.size();
            size_t numOutputs = numInputs == 0 ? 0 : weights2D[0].size();

            weights[i].resize(numInputs * numOutputs);
            for (size_t row = 0; row < numInputs; ++row)
            {
                for (size_t col = 0; col < numOutputs; ++col)
                {
                    weights[i][row * numOutputs + col] = weights2D[row][col];
                }
            }
        }

        return {biases, weights};
    }

    class Network
    {

        public:
            using ProgressCallback = std::function<void(int currentEpoch, int totalEpochs, int currentBatch, int totalBatch, int trainingSample, int trainingSampleNum, float currentLoss)>;

            struct Layer
            {
                size_t numInputs = 0;
                size_t numOutputs = 0;

                //used for forward passes
                vector<float> weights;
                vector<float> biases;

                // allows different activation functions to be used
                function<float(float)> activationFunction;
                function<float(float)> invActivationFunction;

                // used in back propagation to calculate gradients
                vector<float> lastInputs;
                vector<float> lastSums;
                vector<float> lastOutputs;

                // used to store gradiens after back propagation for proper use in the batches and epochs
                vector<float> weightGradients;
                vector<float> biasGradients;

                /// @brief Constructor of the Layer structure. Creates the nodes and weights in the correct size
                /// @param thisLayer number of nodes in this layer 
                /// @param nextLayer number of weights in this layer (number of nodes in the next layer)
                /// @param actvFunc used to store the activation function of each layer. May change to allow different ones in each layer or node later!
                /// @param invActFunc differential of the activation function
                Layer(const size_t numInputs = 0, const size_t numOutputs = 0,
                        const function<float(float)> actvFunc = Softsign,
                        const function<float(float)> invActFunc = InverseSoftsign)
                        : numInputs(numInputs), numOutputs(numOutputs), activationFunction(actvFunc), invActivationFunction(invActFunc)
                {
                    biases.resize(numOutputs, 0.0f);
                    biasGradients.resize(numOutputs, 0.0f);

                    size_t totalWeights = numInputs * numOutputs;
                    weights.resize(totalWeights);
                    weightGradients.resize(totalWeights, 0.0f);

                    for (size_t k = 0; k < totalWeights; ++k)
                    {
                        weights[k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    }
                }

                /// @brief 
                /// @param biasValues 
                /// @param weightValues 
                /// @param actvFunc 
                /// @param invActFunc 
                Layer(const vector<float>& biasValues, const vector<float>& weightValues,
                        const function<float(float)> actvFunc = Softsign,
                        const function<float(float)> invActFunc = InverseSoftsign)
                        : activationFunction(actvFunc), invActivationFunction(invActFunc)
                {
                    biases = biasValues;
                    weights = weightValues;

                    numOutputs = biases.size();
                    numInputs = numOutputs == 0 ? 0 : weights.size() / numOutputs;

                    biasGradients.resize(numOutputs, 0.0f);
                    weightGradients.resize(weights.size(), 0.0f);
                }

                /// @brief Converts the weights and biases into the json structure
                /// @return 
                json CreateJsonLayer() const
                {
                    vector<vector<float>> weights2D(numInputs, vector<float>(numOutputs));
                    for (size_t i = 0; i < numInputs; ++i)
                    {
                        for (size_t j = 0; j < numOutputs; ++j)
                        {
                            weights2D[i][j] = weights[i * numOutputs + j];
                        }
                    }

                    return json{
                        {"Biases", biases},
                        {"Weights", weights2D}
                    };
                }

                /// @brief updates the biases to a deired value. Used to load previous models
                /// @param newBiases 
                void UpdateBiases(const vector<float>& newBiases)
                {
                    biases = newBiases;
                    numOutputs = biases.size();
                    biasGradients.assign(numOutputs, 0.0f);
                }

                /// @brief updates the weights to a deired value. Used to load previous models
                /// @param newWeights 
                void UpdateWeights(const vector<float>& newWeights)
                {
                    weights = newWeights;
                    numInputs = numOutputs == 0 ? 0 : weights.size() / numOutputs;
                    weightGradients.assign(weights.size(), 0.0f);
                }

                /// @brief gets the input to each bias in the next layer
                /// @param inputs the outputs from the last layer
                /// @return the inputs to the next layer
                const vector<float>& ForwardPass(const vector<float>& inputs)
                {
                    if (numOutputs == 0) return inputs;

                    lastInputs = inputs; 
                    lastSums.assign(numOutputs, 0.0f);
                    lastOutputs.resize(numOutputs);

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        lastSums[j] = biases[j];
                    }

                    for (size_t i = 0; i < numInputs; ++i)
                    {
                        float inVal = inputs[i];
                        size_t rowOffset = i * numOutputs;

                        for (size_t j = 0; j < numOutputs; ++j)
                        {
                            lastSums[j] += inVal * weights[rowOffset + j];
                        }
                    }

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        lastOutputs[j] = activationFunction(lastSums[j]);
                    }

                    return lastOutputs;
                }

                /// @brief Adds the changes in gradient calculated in the backpropagation
                /// @param deltas The gradient from the last back propagation
                /// @return The delta from the layer before
                vector<float> AccumulateGradients(const vector<float>& deltas)
                {
                    vector<float> prevDeltas(numInputs, 0.0f);

                    for (size_t i = 0; i < numInputs; ++i)
                    {
                        float inputI = lastInputs[i];
                        size_t rowOffset = i * numOutputs;

                        for (size_t j = 0; j < numOutputs; ++j)
                        {
                            float deltaJ = deltas[j];

                            prevDeltas[i] += weights[rowOffset + j] * deltaJ;
                            weightGradients[rowOffset + j] += inputI * deltaJ;
                        }
                    }

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        biasGradients[j] += deltas[j];
                    }

                    return prevDeltas;
                }

                /// @brief applies the gradient changes to each weight and bias after an epoch
                /// @param learningRate rate of change. Should be different for each epoch
                /// @param currentBatchSize the number items in the epoch (batch size)
                void ApplyGradients(const float learningRate, const size_t currentBatchSize)
                {
                    float invBatch = 1.0f / static_cast<float>(currentBatchSize);
                    size_t totalWeights = weights.size();

                    for (size_t k = 0; k < totalWeights; ++k)
                    {
                        float avgGrad = weightGradients[k] * invBatch;
                        weights[k] -= learningRate * avgGrad;
                        weightGradients[k] = 0.0f;
                    }

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        float avgBiasGrad = biasGradients[j] * invBatch;
                        biases[j] -= learningRate * avgBiasGrad;
                        biasGradients[j] = 0.0f;
                    }
                }
            };

            /// @brief Constructor for the network
            /// @param layerSizes array for the size of each layer. build as { input nodes, hidden nodes 1, ... hidden nodes n, output nodes }
            /// @param actvFunc used to store the activation function of each layer. May change to allow different ones in each layer or node later!
            /// @param invActvFunc the differential of the activation function
            Network(const vector<int> &layerSizes,
                    const function<float(float)> actvFunc = Softsign,
                    const function<float(float)> invActvFunc = InverseSoftsign)
            {
                layers.resize(layerSizes.size() - 1);

                for (size_t i = 0; i < layers.size(); i++)
                {
                    layers[i] = Layer(layerSizes[i], layerSizes[i + 1], actvFunc, invActvFunc); // initialises each layer
                }
            }

            /// @brief Creates a network using the given weights and biases
            /// @param biases vector of each layer's biases
            /// @param weights vector of each layer's weights
            /// @param actvFunc lambda method to act as the activation function
            /// @param invActvFunc lambda method. Must be d/dx of the activation function
            Network(const vector<vector<float>>& biases, const vector<vector<float>>& weights,
                    const function<float(float)> actvFunc = Softsign,
                    const function<float(float)> invActvFunc = InverseSoftsign)
            {
                if (biases.size() != weights.size())
                {
                    throw invalid_argument("Mismatch between number of bias layers and weight layers.");
                }

                layers.reserve(biases.size());
                for (size_t i = 0; i < biases.size(); i++)
                {
                    layers.emplace_back(biases[i], weights[i], actvFunc, invActvFunc);
                }
            }

            Network(){}

            const vector<Layer>& GetLayers() const { return layers; }

            /// @brief shows the shape of the network on a way that can be displayed in the console
            /// @return A string of the network showing its current biases as the nodes, and the weights as the connection between them
            string VisualNetwork() const
            {
                std::ostringstream output;

                for (size_t i = 0; i < layers.size(); ++i)
                {
                    const auto& layer = layers[i];
                    
                    for (size_t inIdx = 0; inIdx < layer.numInputs; ++inIdx)
                    {
                        size_t rowOffset = inIdx * layer.numOutputs;
                        
                        for (size_t outIdx = 0; outIdx < layer.numOutputs; ++outIdx)
                        {
                            output << layer.weights[rowOffset + outIdx] << " ";
                        }

                        if (inIdx != layer.numInputs - 1)
                        {
                            output << "| ";
                        }
                    }
                    output << "\n";

                    for (size_t outIdx = 0; outIdx < layer.numOutputs; ++outIdx)
                    {
                        output << layer.biases[outIdx] << " ";
                    }
                    output << "\n";
                }

                return output.str();
            }

            /// @brief runs through the network to get the final output
            /// @param inputs an array of the inputs to the first layer of the network
            /// @return An array of the output nodes
            const vector<float>& ForwardPass(const vector<float>& inputs)
            {
                if (layers.empty() || inputs.size() != layers[0].numInputs)
                {
                    throw invalid_argument("Input size mismatch with network input layer.");
                }

                const vector<float>* currentInputs = &inputs;
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    // Now works safely because ForwardPass returns a const reference
                    currentInputs = &layers[i].ForwardPass(*currentInputs);
                }
                return *currentInputs;
            }

            /// @brief Trains the neural network using back propagation and gradient descent
            /// @param trainingData array of pairs as {inputs, outputs}
            /// @param epochs number of times it should be trained
            /// @param batchSize number of times the training should take place in each epoch
            /// @param learningRate the rate of change for each epoch
            void Train(const vector<pair<vector<float>, vector<float>>> &trainingData, 
                    int epochs, size_t batchSize, float learningRate, ProgressCallback onProgress = nullptr)
            {
                size_t numSamples = trainingData.size();
                if (numSamples == 0 || batchSize == 0) return;

                float currentLearningRate = learningRate;
                float decayFactor = 0.95f;
                int totalBatches = static_cast<int>((numSamples + batchSize - 1) / batchSize);

                for (int epoch = 0; epoch < epochs; ++epoch)
                {
                    float totalEpochLoss = 0.0f;
                    size_t processedSamples = 0;

                    for (size_t i = 0; i < numSamples; i += batchSize)
                    {
                        size_t currentBatchSize = 0;

                        for (size_t b = 0; b < batchSize && (i + b) < numSamples; b++)
                        {
                            size_t sampleIdx = i + b;
                            const auto& sampleItems = trainingData[sampleIdx];

                            const auto& output = ForwardPass(sampleItems.first);
                            BackPropagation(sampleItems.second);

                            float sampleLoss = 0.0f;
                            for (size_t k = 0; k < output.size(); ++k)
                            {
                                float diff = sampleItems.second[k] - output[k];
                                sampleLoss += diff * diff;
                            }
                            totalEpochLoss += (0.5f * sampleLoss);

                            currentBatchSize++;
                            processedSamples++;
                        }

                        ApplyBatch(currentLearningRate, currentBatchSize);

                        if (onProgress)
                        {
                            int currentBatch = static_cast<int>(i / batchSize) + 1;
                            float averageLoss = totalEpochLoss / processedSamples;
                            onProgress(epoch + 1, epochs, currentBatch, totalBatches, static_cast<int>(processedSamples), static_cast<int>(numSamples), averageLoss);
                        }
                    }

                    // Decay rate updated strictly once per epoch
                    currentLearningRate *= decayFactor;
                }
            }

            json CreateJsonFile() const
            {
                size_t layersSize = layers.size();
                vector<json> jsonLayers(layersSize);

                for (size_t i = 0; i < layersSize; i++)
                {
                    jsonLayers[i] = layers[i].CreateJsonLayer();
                }
                return {
                    {"Layers", jsonLayers}
                };
            }

        private:
            vector<Layer> layers;
        
            /// @brief starts the learning process 'back propagation'. Gets the output from forward pass and the real, ideal output and calculates the gradient descent at each layer
            /// @param idealOutput the desired output for this specific input
            void BackPropagation(const vector<float> &idealOutput)
            {
                const auto &finalOutputs = layers.back().lastOutputs;
                const auto &finalSums = layers.back().lastSums;

                vector<float> deltas(finalOutputs.size());
                for (size_t j = 0; j < finalOutputs.size(); j++)
                {
                    float error = finalOutputs[j] - idealOutput[j];
                    deltas[j] = error * layers.back().invActivationFunction(finalSums[j]);
                }

                for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
                {
                    deltas = layers[i].AccumulateGradients(deltas);

                    if (i > 0)
                    {
                        const auto &prevSums = layers[i - 1].lastSums;
                        for (size_t k = 0; k < deltas.size(); k++)
                        {
                            deltas[k] *= layers[i - 1].invActivationFunction(prevSums[k]);
                        }
                    }
                }
            }

            /// @brief Changes the weights and biases for each layer
            /// @param learningRate rate of change to apply
            /// @param batchSize number of test idems in the batch
            void ApplyBatch(const float learningRate, size_t batchSize)
            {
                for (auto &layer : layers)
                {
                    layer.ApplyGradients(learningRate, batchSize);
                }
            }
    };
};