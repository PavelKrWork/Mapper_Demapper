#include "QAM_mapper.h"
#include "QAM_demapping.h"
#include "structures.h"
#include "external_functions.h"
#include "AWGN_generator.h"
#include <iostream>

constexpr int Nmax = 91;

int main() {
	try {
		auto BitTx = generateBitVector(4800, 0.5);

		Mapper mapping(BitTx);
		auto qpsk_samples = mapping.modulate("QPSK");
		auto qam16_samples = mapping.modulate("QAM16");
		auto qam64_samples = mapping.modulate("QAM64");

		double SNRs[9] = {-20, -15,-10, -5, 0, 5, 10, 15, 20};  // соотношения сигнал-шум [dB]
		// массив с различными дисперсиями шума
		/*double variences[19] = { 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1 };*/
		double variences[Nmax];
		double var = 0.1;
		variences[0] = var;
		for (size_t i = 1; i < Nmax; ++i) {
			variences[i] = var;
			var += 0.01;
		}

		for (size_t i = 0; i < Nmax; ++i) {
			std::cout << variences[i] << ", ";
		}
		std::cout << std::endl;

		std::cout << std::endl;
		for (const auto& SNR : SNRs) {
			std::cout << SNR << std::endl;
			for (const auto& var : variences) {
				AWGN noise(qpsk_samples, SNR, var);
				auto noisedSig = noise.addNoise(qpsk_samples);

				Demapper demapping(noisedSig);
				auto BitRx = demapping.demodulate("QPSK");

				std::cout << std::round(bitErrorProbability(BitTx, BitRx) * 100) / 100 << ", ";
			}

			std::cout << std::endl;
			for (const auto& var : variences) {
				AWGN noise(qam16_samples, SNR, var);
				auto noisedSig = noise.addNoise(qam16_samples);

				Demapper demapping(noisedSig);
				auto BitRx = demapping.demodulate("QAM16");

				std::cout << std::round(bitErrorProbability(BitTx, BitRx) * 100) / 100 << ", ";
			}

			std::cout << std::endl;
			for (const auto& var : variences) {
				AWGN noise(qam64_samples, SNR, var);
				auto noisedSig = noise.addNoise(qam64_samples);

				Demapper demapping(noisedSig);
				auto BitRx = demapping.demodulate("QAM64");

				std::cout << std::round(bitErrorProbability(BitTx, BitRx) * 100) / 100 << ", ";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			
		}
		
	}
	catch (const std::invalid_argument& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}

	catch (const std::exception& e) {
		std::cerr << "Unexpected exception: " << e.what() << std::endl;
	}

	catch (...) {
		std::cerr << "An unknown error: " << std::endl;
	}

	return 0;
}