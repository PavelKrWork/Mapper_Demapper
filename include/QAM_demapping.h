#pragma once
#include "structures.h"
#include "external_functions.h"
#include <vector>

class Demapper {
public:
	Demapper (const std::vector<IQ>& RecievedIQ) {
		IQSamples = RecievedIQ;
	}

	BitArray demodulate(const char* ConstType) {
        BitArray res;

        std::vector<BitArray> bits;
        std::vector<IQ> constellation;
        int bitsPerSymbol = 0;

        if (ConstType == "QPSK") {
            constellation = QPSK_TABLE;
            bitsPerSymbol = 2;
            bits = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        }

        else if (ConstType == "QAM16") {
            constellation = QAM16_TABLE;
            bitsPerSymbol = 4;
            bits = { {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
                    {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
                    {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
                    {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1} };
        }

        else if (ConstType == "QAM64") {
            constellation = QAM64_TABLE;
            bitsPerSymbol = 6;
            bits = { {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 1},
                     {0, 0, 0, 1, 0, 0}, {0, 0, 0, 1, 0, 1}, {0, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 1},
                     {0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 1, 1},
                     {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1},
                     {0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 1, 0, 0, 1, 0}, {0, 1, 0, 0, 1, 1},
                     {0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 1}, {0, 1, 0, 1, 1, 0}, {0, 1, 0, 1, 1, 1},
                     {0, 1, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 1}, {0, 1, 1, 0, 1, 0}, {0, 1, 1, 0, 1, 1},
                     {0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1}, {0, 1, 1, 1, 1, 0}, {0, 1, 1, 1, 1, 1},
                     {1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 1},
                     {1, 0, 0, 1, 0, 0}, {1, 0, 0, 1, 0, 1}, {1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 1},
                     {1, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 1}, {1, 0, 1, 0, 1, 0}, {1, 0, 1, 0, 1, 1},
                     {1, 0, 1, 1, 0, 0}, {1, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 1},
                     {1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 1, 0}, {1, 1, 0, 0, 1, 1},
                     {1, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {1, 1, 0, 1, 1, 1},
                     {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 1},
                     {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1} };
        }

        for (const auto& symbol : IQSamples) {
            // найти ближайшую точку в созвездии
            auto nearest = nearestPoint(constellation, symbol);

            // индекс ближайшей точки
            int index = 0;
            for (size_t i = 0; i < constellation.size(); ++i) {
                if (nearest.Im == constellation[i].Im && nearest.Re == constellation[i].Re) index = i;
            }

            for (size_t i = 0; i < bits[index].size(); ++i) {
                res.push_back(bits[index][i]);
            }
            
        }

        return res;
	}
private:
	std::vector<IQ> IQSamples;
};