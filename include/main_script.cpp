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

		std::vector<int> SNRs(2001);  // соотношения сигнал-шум [dB]
		double SNR = -20;
		for (unsigned int i = 0; i < SNRs.size(); ++i) {
			std::cout << SNR << ", ";
			SNRs[i] = SNR;
			SNR += 0.02;
		}
		std::cout << std::endl;

		std::vector<double> BERS1(SNRs.size());
		for (unsigned int i = 0; i < SNRs.size(); ++i) {
			AWGN noise(qpsk_samples, SNRs[i]);
			auto noised_sig = noise.addNoise(qpsk_samples);
			Demapper demap(noised_sig);
			auto BitRx = demap.demodulate("QPSK");
			auto ber = bitErrorProbability(BitTx, BitRx);
			std::cout << ber << ", ";
			BERS1[i] = ber;
		}

		std::cout << std::endl;

		std::vector<double> BERS2(SNRs.size());
		for (unsigned int i = 0; i < SNRs.size(); ++i) {
			AWGN noise(qam16_samples, SNRs[i]);
			auto noised_sig = noise.addNoise(qam16_samples);
			Demapper demap(noised_sig);
			auto BitRx = demap.demodulate("QAM16");
			auto ber = bitErrorProbability(BitTx, BitRx);
			std::cout << ber << ", ";
			BERS2[i] = ber;
		}

		std::cout << std::endl;
		std::vector<double> BERS3(SNRs.size());
		for (unsigned int i = 0; i < SNRs.size(); ++i) {
			AWGN noise(qam64_samples, SNRs[i]);
			auto noised_sig = noise.addNoise(qam64_samples);
			Demapper demap(noised_sig);
			auto BitRx = demap.demodulate("QAM64");
			auto ber = bitErrorProbability(BitTx, BitRx);
			std::cout << ber << ", ";
			BERS3[i] = ber;
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