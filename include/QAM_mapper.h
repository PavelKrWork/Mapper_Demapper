#pragma once
#include <vector>
#include <stdexcept>
#include "structures.h"


class Mapper {
public:
	Mapper(const BitArray& BitSig) {
		BitSignal = BitSig;
	}

	std::vector<IQ> modulate(const char* ConstType) {
		std::vector<IQ> IQ_samples;

		if (ConstType == "QPSK") {
			if (BitSignal.size() % 2 != 0) throw std::invalid_argument("Number of bits should be multiple of 2 for QPSK.");
			for (size_t i = 0; i < BitSignal.size(); i += 2) {
				int index = BitSignal[i] * 2 + BitSignal[i + 1];
				IQ_samples.push_back(QPSK_TABLE[index]);
			}
		}

		else if (ConstType == "QAM16") {
			if (BitSignal.size() % 4 != 0) throw std::invalid_argument("Number of bits should be multiple of 4 for QAM16.");
			for (size_t i = 0; i < BitSignal.size(); i += 4) {
				int index = BitSignal[i] * 8 + BitSignal[i + 1] * 4 + BitSignal[i + 2] * 2 + BitSignal[i + 3];
				IQ_samples.push_back(QAM16_TABLE[index]);
			}
		}

		else if (ConstType == "QAM64") {
			if (BitSignal.size() % 6 != 0) throw std::invalid_argument("Number of bits should be multiple of 6 for QAM64.");
			for (size_t i = 0; i < BitSignal.size(); i += 6) {
				int index = BitSignal[i] * 32 + BitSignal[i + 1] * 16 + BitSignal[i + 2] * 8 + BitSignal[i + 3] * 4 + BitSignal[i + 4] * 2 + BitSignal[i + 5];
				IQ_samples.push_back(QAM64_TABLE[index]);
			}
		}

		else {
			throw std::invalid_argument("Unsupported modulation type.");
		}

		return IQ_samples;
		
	}

private:
	BitArray BitSignal;
};