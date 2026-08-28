#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace Network
{
    using json = nlohmann::json;
    using std::vector;
    using std::pair;
    using std::string;
    using std::to_string;
    using std::function;
    using std::invalid_argument;

    double Softsign(double inputValue) // Essentially sigmoid but faster
    {
        return inputValue / (1.0 + abs(inputValue));
    }

    double InverseSoftsign(double inputValue)
    {
        double denom = 1.0 + abs(inputValue);
        return 1.0 / (denom * denom);
    }

    /// @brief Converts the json file into a pair of each layers's weights and biases
    /// @param weightsFile json object of the weights and biases file
    /// @return Layer's weights and biases
    pair<vector<vector<double>>, vector<vector<vector<double>>>> JsonToVectors(const json& weightsFile)
    {
        size_t numLayers = weightsFile["Layers"].size();

        vector<vector<double>> biases(numLayers);
        vector<vector<vector<double>>> weights(numLayers);

        for (size_t i = 0; i < numLayers; i++)
        {
            biases[i] = weightsFile["Layers"][i]["Biases"].get<vector<double>>();
            weights[i] = weightsFile["Layers"][i]["Weights"].get<vector<vector<double>>>();
        }

        return {biases, weights};
    }

    class Network
    {

        public:
            using ProgressCallback = std::function<void(int currentEpoch, int totalEpochs, int currentBatch, int totalBatch, int trainingSample, int trainingSampleNum, double currentLoss)>;

            struct Layer
            {
                //used for forward passes
                vector<vector<double>> weights;
                vector<double> biases;

                // allows different activation functions to be used
                function<double(double)> activationFunction;
                function<double(double)> invActivationFunction;

                // used in back propagation to calculate gradients
                vector<double> lastInputs;
                vector<double> lastSums;
                vector<double> lastOutputs;

                // used to store gradiens after back propagation for proper use in the batches and epochs
                vector<vector<double>> weightGradients;
                vector<double> biasGradients;

                /// @brief Constructor of the Layer structure. Creates the nodes and weights in the correct size
                /// @param thisLayer number of nodes in this layer 
                /// @param nextLayer number of weights in this layer (number of nodes in the next layer)
                /// @param actvFunc used to store the activation function of each layer. May change to allow different ones in each layer or node later!
                /// @param invActFunc differential of the activation function
                Layer(int thisLayer = 0, int nextLayer = 0,
                    function<double(double)> actvFunc = Softsign,
                    function<double(double)> invActFunc = InverseSoftsign)
                {
                    activationFunction = actvFunc;
                    invActivationFunction = invActFunc;

                    biases.resize(nextLayer, 0.0);
                    biasGradients.resize(nextLayer, 0.0);

                    weights.resize(thisLayer);
                    weightGradients.resize(thisLayer, vector<double>(nextLayer, 0.0));

                    for (vector<double> &weightFrom : weights)
                    {
                        weightFrom.resize(nextLayer);
                        for (double &weightTo : weightFrom)
                        {
                            weightTo = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // initialize the weights as random doubles
                        }
                    }
                }

                Layer(vector<double> biasValues, vector<vector<double>> weightValues,
                    function<double(double)> actvFunc = Softsign,
                    function<double(double)> invActFunc = InverseSoftsign
                    )
                {
                    activationFunction = actvFunc;
                    invActivationFunction = invActFunc;

                    biases = biasValues;
                    weights = weightValues;

                    // Match gradient accumulators to loaded layout
                    size_t numInputs = weights.size();
                    size_t numOutputs = weights.empty() ? 0 : weights[0].size();

                    biasGradients.resize(biases.size(), 0.0);
                    weightGradients.resize(numInputs, vector<double>(numOutputs, 0.0));
                }

                /// @brief Converts the weights and biases into the json structure
                /// @return 
                json CreateJsonLayer()
                {
                    json jsonLayer = {
                        {"Biases", biases},
                        {"Weights", weights}
                    };
                    return jsonLayer;
                }

                /// @brief updates the biases to a deired value. Used to load previous models
                /// @param newBiases 
                void UpdateBiases(vector<double> newBiases)
                {
                    biases = newBiases;
                    biasGradients.assign(biases.size(), 0.0);
                }

                /// @brief updates the weights to a deired value. Used to load previous models
                /// @param newWeights 
                void UpdateWeights(vector<vector<double>> newWeights)
                {
                    weights = newWeights;
                    size_t numInputs = weights.size();
                    size_t numOutputs = weights.empty() ? 0 : weights[0].size();
                    weightGradients.assign(numInputs, vector<double>(numOutputs, 0.0));
                }

                /// @brief gets the input to each bias in the next layer
                /// @param inputs the outputs from the last layer
                /// @return the inputs to the next layer
                vector<double> ForwardPass(vector<double> &inputs)
                {
                    if (weights.empty() || weights[0].empty()) // if it is the final layer
                        return inputs;

                    lastInputs = inputs; // hold onto for the back propagation
                    size_t numInputs = weights.size();
                    size_t numOutputs = weights[0].size();

                    lastSums.assign(numOutputs, 0.0);
                    lastOutputs.assign(numOutputs, 0.0);

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        double sum = biases[j];
                        for (size_t i = 0; i < numInputs; ++i)
                        {
                            sum += inputs[i] * weights[i][j]; // function a = ∑(w*a-1) + b
                        }
                        lastSums[j] = sum; // used in back propagation
                        lastOutputs[j] = activationFunction(sum); // used in back propagation
                    }

                    return lastOutputs;
                }

                /// @brief Adds the changes in gradient calculated in the backpropagation
                /// @param deltas The gradient from the last back propagation
                /// @return The delta from the layer before
                vector<double> AccumulateGradients(const vector<double> &deltas)
                {
                    size_t numInputs = weights.size();
                    size_t numOutputs = weights[0].size();

                    vector<double> prevDeltas(numInputs, 0.0);

                    for (size_t j = 0; j < numOutputs; j++)
                    {
                        double deltaJ = deltas[j];

                        for (size_t i = 0; i < numInputs; i++)
                        {
                            prevDeltas[i] += weights[i][j] * deltaJ;

                            weightGradients[i][j] += lastInputs[i] * deltaJ;
                        }

                        biasGradients[j] += deltaJ;
                    }

                    return prevDeltas;
                }

                /// @brief applies the gradient changes to each weight and bias after an epoch
                /// @param learningRate rate of change. Should be different for each epoch
                /// @param currentBatchSize the number items in the epoch (batch size)
                void ApplyGradients(double learningRate, size_t currentBatchSize)
                {
                    size_t numInputs = weights.size();
                    size_t numOutputs = weights[0].size();

                    for (size_t j = 0; j < numOutputs; j++)
                    {
                        for (size_t i = 0; i < numInputs; i++)
                        {
                            double avgGrad = weightGradients[i][j] / currentBatchSize;
                            weights[i][j] -= learningRate * avgGrad;
                            weightGradients[i][j] = 0.0;
                        }

                        double avgBiasGrad = biasGradients[j] / currentBatchSize;
                        biases[j] -= learningRate * avgBiasGrad;
                        biasGradients[j] = 0.0;
                    }
                }
            };

            /// @brief Constructor for the network
            /// @param layerSizes array for the size of each layer. build as { input nodes, hidden nodes 1, ... hidden nodes n, output nodes }
            /// @param actvFunc used to store the activation function of each layer. May change to allow different ones in each layer or node later!
            /// @param invActvFunc the differential of the activation function
            Network(vector<int> &layerSizes,
                    function<double(double)> actvFunc = Softsign,
                    function<double(double)> invActvFunc = InverseSoftsign)
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
            Network(vector<vector<double>> biases, vector<vector<vector<double>>> weights,
                    function<double(double)> actvFunc = Softsign,
                    function<double(double)> invActvFunc = InverseSoftsign)
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
            string VisualNetwork()
            {
                string output = "";
                for (size_t i = 0; i < layers.size(); i++)
                {
                    for (size_t j = 0; j < layers[i].weights.size(); j++)
                    {
                        for (size_t k = 0; k < layers[i].weights[j].size(); k++)
                        {
                            output += to_string(layers[i].weights[j][k]) + " ";
                        }
                        output += j != layers[i].weights.size() - 1 ? "| " : "";
                    }
                    output += "\n";
                    for (size_t j = 0; j < layers[i].biases.size(); j++)
                    {
                        output += to_string(layers[i].biases[j]) + " ";
                    }
                    output += "\n";
                }

                return output;
            }

            /// @brief runs through the network to get the final output
            /// @param inputs an array of the inputs to the first layer of the network
            /// @return An array of the output nodes
            vector<double> ForwardPass(vector<double> inputs)
            {
                if (inputs.size() != layers[0].weights.size())
                {
                    throw invalid_argument("Input size mismatch with network input layer.");
                }

                for (size_t i = 0; i < layers.size(); ++i)
                {
                    inputs = layers[i].ForwardPass(inputs); // calculates each layer and outputs the array from the other side
                }
                return inputs;
            }

            /// @brief Trains the neural network using back propagation and gradient descent
            /// @param trainingData array of pairs as {inputs, outputs}
            /// @param epochs number of times it should be trained
            /// @param batchSize number of times the training should take place in each epoch
            /// @param learningRate the rate of change for each epoch
            void Train(const vector<pair<vector<double>, vector<double>>> &trainingData, 
                    int epochs, 
                    size_t batchSize, 
                    double learningRate,
                    ProgressCallback onProgress = nullptr)
            {
                size_t numSamples = trainingData.size();

                for (int epoch = 0; epoch < epochs; ++epoch)
                {
                    double totalEpochLoss = 0.0;
                    size_t processedSamples = 0;

                    double learnRateTemp = learningRate;
                    double decayFactor = 0.95;
                    for (size_t i = 0; i < numSamples; i += batchSize)
                    {
                        size_t currentBatchSize = 0;

                        for (size_t b = 0; b < batchSize && (i + b) < numSamples; b++)
                        {
                            size_t sampleIdx = i + b;

                            pair<vector<double>, vector<double>> sampleItems = trainingData[sampleIdx];
                            auto output = ForwardPass(sampleItems.first);
                            BackPropagation(sampleItems.second);

                            double sampleLoss = 0.0;
                            for (size_t k = 0; k < output.size(); ++k)
                            {
                                double diff = sampleItems.second[k] - output[k];
                                sampleLoss += diff * diff;
                            }
                            totalEpochLoss += (0.5 * sampleLoss);

                            currentBatchSize++;
                            processedSamples++;
                            if (onProgress)
                            {
                                onProgress(epoch + 1, epochs, i / batchSize + 1, (numSamples + batchSize - 1) / batchSize, i + b + 1, trainingData.size(), totalEpochLoss / processedSamples);
                            }
                        }
                        ApplyBatch(learnRateTemp, currentBatchSize);
                        learnRateTemp *= decayFactor;
                    }
                }
            }

            json CreateJsonFile()
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
            void BackPropagation(const vector<double> &idealOutput)
            {
                const auto &finalOutputs = layers.back().lastOutputs;
                const auto &finalSums = layers.back().lastSums;

                vector<double> deltas(finalOutputs.size());
                for (size_t j = 0; j < finalOutputs.size(); j++)
                {
                    double error = finalOutputs[j] - idealOutput[j];
                    deltas[j] = error * layers.back().invActivationFunction(finalSums[j]);
                }

                for (int i = layers.size() - 1; i >= 0; i--)
                {
                    deltas = layers[i].AccumulateGradients(deltas);

                    if (i > 0)
                    {
                        for (size_t k = 0; k < deltas.size(); k++)
                        {
                            deltas[k] *= layers[i - 1].invActivationFunction(layers[i - 1].lastSums[k]);
                        }
                    }
                }
            }

            /// @brief Changes the weights and biases for each layer
            /// @param learningRate rate of change to apply
            /// @param batchSize number of test idems in the batch
            void ApplyBatch(double learningRate, size_t batchSize)
            {
                for (auto &layer : layers)
                {
                    layer.ApplyGradients(learningRate, batchSize);
                }
            }
    };
};