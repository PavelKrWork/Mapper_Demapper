#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"
#include "int_check_large.h"

using namespace matlab::data;
using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
public:

    ArrayFactory factory;

    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        std::cout<<"Simulation has started"<<std::endl;

        int ue_num                  = inputs[2][0];
        int threads                 = inputs[3][0];
        int n_rx                    = inputs[4][0];
        int n_tti                   = inputs[5][0];
        int n_sc                    = inputs[6][0];
        int n_tx                    = inputs[7][0];
        int quantize_rx             = inputs[8][0];
        int quantize_tx             = inputs[9][0];
        int quantize_sc             = inputs[10][0];
        int quantize_tti            = inputs[11][0];
        int tti_order               = inputs[12][0];
        int sc_format               = inputs[13][0];
        int tx_format               = inputs[14][0];
        int rx_format               = inputs[15][0];
        int rank                    = inputs[16][0];
        int tti_cut                 = inputs[17][0];
        int n_tx_onedim_1           = inputs[18][0];
        int n_tx_onedim_2           = inputs[19][0];
        int firstALS_iterations     = inputs[20][0];
        int tailALS_iterations      = inputs[21][0];
        int extfft_multiplier       = inputs[22][0];
        double extfft_resolution    = inputs[23][0];
        double SNR                  = inputs[24][0];

        TypedArray<std::complex<double>> chan_ml = std::move(inputs[0]);
        TypedArray<std::complex<double>> chan_ml_2 = std::move(inputs[1]);

        std::complex<double> ****chan = new std::complex<double>***[n_rx];
        std::complex<double> ****chan_2 = new std::complex<double>***[n_rx];

        std::complex<double> ****results = new std::complex<double>***[n_rx];


        for (int rx = 0; rx < n_rx; rx++) {
            chan[rx] = new std::complex<double>**[n_tti];
            chan_2[rx] = new std::complex<double>**[n_tti];

            results[rx] = new std::complex<double>**[n_tti];
            for (int tti = 0; tti < n_tti; tti++) {
                chan[rx][tti] = new std::complex<double>*[n_sc];
                chan_2[rx][tti] = new std::complex<double>*[n_sc];

                results[rx][tti] = new std::complex<double>*[n_sc];
                for (int sc = 0; sc < n_sc; sc++) {
                    chan[rx][tti][sc] = new std::complex<double>[n_tx];
                    chan_2[rx][tti][sc] = new std::complex<double>[n_tx];

                    results[rx][tti][sc] = new std::complex<double>[n_tx];
                    for (int tx = 0; tx < n_tx; tx++) {
                        chan[rx][tti][sc][tx] = std::complex<double>(chan_ml[rx][tti][sc][tx]);

                        chan_2[rx][tti][sc][tx] = std::complex<double>(chan_ml_2[rx][tti][sc][tx]);
                    };
                };
            };
        };

        results = simulate(chan, chan_2, ue_num, threads, n_rx, n_tti, n_sc, n_tx, quantize_rx, quantize_tx, quantize_sc, quantize_tti, tti_order, sc_format, tx_format, rx_format, rank, tti_cut, n_tx_onedim_1, n_tx_onedim_2, firstALS_iterations, tailALS_iterations, extfft_multiplier, extfft_resolution, SNR);
	    TypedArray<std::complex<double>> results_ml = factory.createArray<std::complex<double>>({n_rx,n_tti,n_sc,n_tx});

        for (int rx = 0; rx < n_rx; rx++) {
            for (int tti = 0; tti < n_tti; tti++) {
                for (int sc = 0; sc < n_sc; sc++) {
                    for (int tx = 0; tx < n_tx; tx++) {
                        results_ml[rx][tti][sc][tx] = results[rx][tti][sc][tx];
//                        if((rx==0) && (tti == 0) && (sc == 0) ){
//                            std::cout<<results[rx][tti][sc][tx].real()<<std::endl;
//                        }
                    };
                };
            };
        };

        outputs[0] = results_ml;

        std::cout<<"Simulation has finished"<<std::endl;
    }
};
