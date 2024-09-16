#pragma once
#include <random>
#include <cmath>
#include "structures.h"
#include "external_functions.h"

class AWGN {
public:
	AWGN(const std::vector<IQ>& signal, double SNR, double var) {
		std::random_device rd;
		std::mt19937 gen(rd());

		double P_sig = powerSignal(signal);
		double A_noise = sqrt(P_sig / pow(10, SNR / 10));

		std::vector<IQ> AWGN_noise(signal.size());

		std::normal_distribution<double> dist(0, var);
		for (int i = 0; i < signal.size(); ++i) {
			double I = A_noise * dist(gen);
			double Q = A_noise * dist(gen);
			IQ iq;
			iq.Re = I;
			iq.Im = Q;
			AWGN_noise[i] = iq;
		}
		noise = AWGN_noise;
	}

	std::vector<IQ> addNoise(const std::vector<IQ> &signal) {
		std::vector<IQ> result(signal.size());
		for (size_t i = 0; i < signal.size(); ++i) {
			result[i].Re = signal[i].Re + noise[i].Re;
			result[i].Im = signal[i].Im + noise[i].Im;
		}

		return result;
	}

private:
	std::vector<IQ> noise;

};