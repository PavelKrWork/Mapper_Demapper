#pragma once
#include <vector>
#include <iostream>
#include <random>
#include "structures.h"

void printIQ(const std::vector<IQ>& v) {
	for (int i = 0; i < v.size(); ++i) {
		std::cout << v[i].Re << " " << v[i].Im << std::endl;
	}
	std::cout << std::endl;
}

double powerSignal(const std::vector<IQ>& signal) {
	double mean_ampl = 0;
	for (size_t i = 0; i < signal.size(); ++i) {
		double sig_ampl = sqrt(signal[i].Im * signal[i].Im + signal[i].Re * signal[i].Re);
		mean_ampl += sig_ampl;
	}
	mean_ampl /= signal.size();

	return mean_ampl;
}

std::vector<bool> generateBitVector(size_t size, double probability) {
	std::vector<bool> bitVector(size);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution d(probability);

	for (auto bit : bitVector) {
		bit = d(gen);
	}

	return bitVector;
}

// Нахождение ближайшей точки в созвездии
IQ nearestPoint(const std::vector<IQ>& constellation, const IQ& point) {
    double minDistance = sqrt(pow(point.Re - constellation[0].Re, 2) + pow(point.Im - constellation[0].Im, 2));
    IQ nearest = constellation[0];

    for (const auto& c : constellation) {
        double distance = sqrt(pow(point.Re - c.Re, 2) + pow(point.Im - c.Im, 2));
        if (distance < minDistance) {
            minDistance = distance;
            nearest = c;
        }
    }

    return nearest;
}

// Подсчет вероятности ошибки на бит
double bitErrorProbability(BitArray BitTx, BitArray BitRx) {
	if (BitTx.size() != BitRx.size()) {
		throw std::invalid_argument("The signal lengths are not equal.");
	}
	double countErrorBits = 0;
	for (size_t i = 0; i < BitTx.size(); ++i) {
		if (BitTx[i] != BitRx[i]) countErrorBits++;
	}
	double probability = countErrorBits / BitTx.size();
	return  probability;
}