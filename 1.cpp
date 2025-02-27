#include <algorithm>
#include <vector>
#include <omp.h>

#include "rainbow.h"
#include "CanonicalTensor.hpp"
#include "FullTensor.hpp"

#define NUM_LEVELS_COMMON 32 // 4096 // common coefficient phase discretization

#define NUM_LEVELS_RX 512 // 4096 // rx phase discretization, 2^b
#define NUM_LEVELS_TX 512 // 4096 // tx phase discretization, 2^b
#define NUM_LEVELS_SC 512 // 4096 // sc phase discretization, 2^b
#define NUM_LEVELS_TTI 512 // 4096 // tti phase discretization, 2^b

// another float from range [A,B] with NUMLEVELS:
#define NUM_LEVELS_AMP_FROM_RANGE 512
#define NUM_LEVELS_AMP_ORDER1 512 // 4096 // for discretization, 2^b, amplitudes
#define NUM_LEVELS_AMP_ORDER2 512 // 4096 // for discretization, 2^b, 2-nd order coefs

// steering approximations thresholds, used only when save_flag==true
#define AMPL_THRESHOLD 0.1  // nonnegative number from 0 to INF, recommended ~ 0.1-0.2
#define PHASE_THRESHOLD 0.95  // nonnegative number from 0 to 1, recommended ~ 0.8-0.95

struct FFTvalue {
	double tx_phase_1;
	double tx_phase_2;
	double sc_phase;
	double value;
};

int FFTCompare(const void* f1, const void* f2) { return (((FFTvalue*)f1)->value < ((FFTvalue*)f2)->value); };

double rounder(double x, int num_of_levels)
{
	// process the sign:
	double tmp;
	tmp=x;
	if (x>M_PI) {tmp=x-2*M_PI;}
	if (x<-M_PI) {tmp=x+2*M_PI;}
	// now tmp is from range [-M_PI,M_PI]
	double step=M_PI/((num_of_levels-1)*1.0);
	return (std::round(tmp/step)*step);
};

double rounder_amp(double x)
{
	// process the absolute value:
	double tmp;
	tmp=x;
	if (std::abs(x)>1.0) {tmp=1.0*(x/std::abs(x));}  //<-- make it not greater than 1.0
	double step=1.0/((NUM_LEVELS_AMP_ORDER2-1)*1.0);
	return (std::round(tmp/step)*step);
};

double simple_positive_flt_bit_dec(double x)   // only positive values (amplitudes, absolute or relative)
{
	double step= 1.0 / (NUM_LEVELS_AMP_ORDER1 - 1);
	double result;
	double tmp;
	if (x>=1.0) {
		tmp = 1.0/x;
		if ((std::round(tmp/step))>0) {
			result = 1.0/((std::round(tmp/step)*step));
		} else {
			result = (NUM_LEVELS_AMP_ORDER1-1);
		}
	} else {
		result = (std::round(x/step)*step);
	}
	return result;
}

// artificial negative floating point: (*sign*) (0.*) * 10^(+ p)
double art_flt(double x, int NUM_LEVELS_SIGNIFICAND, int NUM_LEVELS_EXPONENT)
{
	double step=1.0/((NUM_LEVELS_SIGNIFICAND)*1.0);
	double result;
	double tmp=std::abs(x);
	int p=0;
	while ((tmp>1.0)&&(p<NUM_LEVELS_EXPONENT))
	{
		tmp=tmp/10.0;
		p += 1;
	}
	// while ((tmp<step)&&(tmp<0.1)&&(std::abs(p)<NUM_LEVELS_EXPONENT))
	// {
	// 	tmp=tmp*10.0;
	// 	p -= 1;
	// }	
	if (tmp<=1.0) {
		result = (std::round(tmp/step)*step);
		if (std::round(tmp/step)==0) {result = step;}
	} else {
		result = 1.0;
	}
	// multiply significand and exponent
	for (int i=0; i<std::abs(p); i++) {
		if (p>0) {
			result = result * 10;
		} else {
			result = result / (double) 10;
		}
	}
	printf("Art float %f %f\n", std::copysign(result, x), x);
	return std::copysign(result, x);
}

// descretize x from [A,B] with NUMLEVELS
double flt_by_range(double x, double A, double B)
{
	double tmp;
	tmp=x;
	bool flag=false;
	if (x>B) {
		tmp=B;
		printf("Real value out of range [%f,%f]: %f\n", A, B, x);
		flag=true;
	}
	if (x<A) {
		tmp=A;
		printf("Real value out of range [%f,%f]: %f\n", A, B, x);
		flag=true;
	}
	// now tmp is from range [A,B]
	double step=(B-A)/((NUM_LEVELS_AMP_FROM_RANGE-1)*1.0);
	tmp=(std::round(tmp/step)*step);
	if (flag) { tmp=art_flt(x,NUM_LEVELS_AMP_FROM_RANGE/4,2); }
	return tmp;
};

int account_bits_phase_save(int num_tx_1, int num_tx_2, int num_sc, int num_phases_sc, int num_phases_tx1, int num_phases_tx2, int num_amplitudes)
{
	int NUM_BITS_TX=std::ceil(std::log2(NUM_LEVELS_TX));
	int NUM_BITS_SC=std::ceil(std::log2(NUM_LEVELS_SC));
	int NUM_BITS_AMP=std::ceil(std::log2(NUM_LEVELS_AMP_ORDER2));
	int bits_add=0;
	bits_add += num_phases_sc*(num_sc-1)*(NUM_BITS_SC+1);
	bits_add += num_phases_tx1*(num_tx_1-1)*(NUM_BITS_TX+1);
	bits_add += num_phases_tx2*(num_tx_2-1)*(NUM_BITS_TX+1);
	bits_add += num_amplitudes*(NUM_BITS_AMP+1);
	return bits_add;
}

float bit_calculator(const int * n_tx_twodim, const int * rank, const int * tti_cut, const int * tx_order, const int * tti_order, 
		int rx_format, int tx_format, int sc_format, int num_rx, int num_tx, int num_sc, int num_tti, int bits_add, int num_chunks)
{
	int chunk_size=0;
	int abs_flt=std::ceil(std::log2(NUM_LEVELS_AMP_ORDER1))+1;  //1 bit to define >1 or <1
	int abs_flt_range=std::ceil(std::log2(NUM_LEVELS_AMP_FROM_RANGE));
	// int art_flt= 1 + std::ceil(std::log2(NUM_LEVELS_SIGNIFICAND)) + 1 + std::ceil(std::log2(NUM_LEVELS_EXPONENT));
	int NUM_BITS_COMMON=std::ceil(std::log2(NUM_LEVELS_COMMON));
	int NUM_BITS_RX=std::ceil(std::log2(NUM_LEVELS_RX));
	int NUM_BITS_TX=std::ceil(std::log2(NUM_LEVELS_TX));
	int NUM_BITS_SC=std::ceil(std::log2(NUM_LEVELS_SC));
	int NUM_BITS_TTI=std::ceil(std::log2(NUM_LEVELS_TTI));
	int NUM_BITS_AMP=std::ceil(std::log2(NUM_LEVELS_AMP_ORDER2));
	float num_bits_per_tti;
	if (rx_format==1){
		chunk_size=chunk_size+(NUM_BITS_RX+1);  // <--phase decline with its sign
	} else {
		chunk_size=chunk_size+(abs_flt+NUM_BITS_RX+1)*(num_rx);  // <--phase decline with its sign
	}
	if (tx_format==1){
		chunk_size=chunk_size+2*(NUM_BITS_TX+1)+2*(NUM_BITS_AMP+1)*(*tx_order-1);  // <-- tx1d, tx2d phase declines with signs
	} else {
		chunk_size=chunk_size+(abs_flt+NUM_BITS_TX+1)*(num_tx);
	}
	if (sc_format==1){
		chunk_size=chunk_size+(NUM_BITS_SC+1);  // <--phase decline with its sign
	} else {
		chunk_size=chunk_size+(abs_flt+NUM_BITS_SC+1)*(num_sc);
	}
	if (*tti_order>0){
		chunk_size=chunk_size+(NUM_BITS_TTI+1)+(NUM_BITS_AMP+1)*(*tti_order-1);  // <--phase decline with its sign and amplitude polynomial coefficients
	} else {
		chunk_size=chunk_size+(abs_flt+NUM_BITS_TTI+1)*(num_tti);
	}
	chunk_size=chunk_size+abs_flt_range+(NUM_BITS_COMMON+1);  //<--common coefficients
	num_bits_per_tti=(*n_tx_twodim)*(*rank)*chunk_size/ (float) (*tti_cut) + bits_add/ (float) ((*tti_cut)*num_chunks);
	return num_bits_per_tti;
}

void take_out_coef_noapp(zlx * iappr, const int * iactr0, const int * splash, const int * splash_size, const int * rank, bool quant_flag, int num_levs, double * out_coef1, double * out_coef2)
{
	int iactr, dimmlt;
	zlx zone = 1.0;
	zlx zzero = 0.0;

	iactr=*iactr0;
	dimmlt = 1.0;
	zlx * complex_coef = new zlx[*rank];
	for(int r=0; r<*rank; r++) {complex_coef[r]=zone;}

	for(int d = 0; d < *splash_size; d++)
	{
		for(int r = 0; r < *rank; r++)
		{
			zlx first_el=zone;
			if (std::abs(iappr[iactr])>0.0001) {
				first_el=iappr[iactr];
			} 
			complex_coef[r]=complex_coef[r]*first_el;
			for(int i = 0; i < splash[d]; i++)
			{
				iappr[iactr]=iappr[iactr]/first_el;
				if (quant_flag) {
					double phase=rounder(std::arg(iappr[iactr]), num_levs);
					double absolute=std::abs(iappr[iactr]);
					absolute = simple_positive_flt_bit_dec(absolute);
					iappr[iactr].real(absolute*cos(phase));
					iappr[iactr].imag(absolute*sin(phase));
				}
				iactr++;
			};
		};
		dimmlt *= splash[d];
	};
	
	// output arrays:
	for(int r=0; r<*rank; r++) 
	{
		out_coef1[r]=std::log(std::abs(complex_coef[r]));
		out_coef2[r]=std::arg(complex_coef[r]);			
	}

	delete[] complex_coef;
}

void multiply_factor_common_coef(zlx * iappr, const int * iactr0, const int * splash, const int * splash_size, const int * rank, const float * out_coef1, const float * out_coef2)
{
	int iactr, dimmlt;
	zlx zone = 1.0;
	zlx zzero = 0.0;

	iactr=*iactr0;
	dimmlt = 1.0;
	zlx tempo;
	double otmp;

	for(int d = 0; d < *splash_size; d++)
	{
		for(int r = 0; r < *rank; r++)
		{
			tempo.real(cos(out_coef2[r]));
			tempo.imag(sin(out_coef2[r]));
			otmp=out_coef1[r];
			for(int i = 0; i < splash[d]; i++)
			{
				if (d==0) {iappr[iactr]=iappr[iactr]*tempo*std::exp(otmp);}
				iactr++;
			};
		};
		dimmlt *= splash[d];
	};
}

// recursive algorithm to find phases
// output : sum_1, sum_2, sum_3, sum_4 (to calculate k, b, y=kx+b), sum_norm
// added : ampl_arr, phase_arr, - output arrays with amplitude and phase of factor, to make procedure more flexible
void cycle_dim(const int* dim, zlx P, const int* offset, int r, int* multiindex, int* N_dim, int* k_pi, double* sum_norm, double* sum_1, double* sum_2,
		double* sum_3, double* sum_4, double* phase_m1, const zlx* factors, double* ampl_arr, double* phase_arr) {
	int N_iter = dim[(*N_dim)];
	for (int i = 0; i < N_iter; i++) {
		zlx P_new = P * factors[offset[(*N_dim)] + i + r * dim[(*N_dim)]];  // multiplication of quantized factors
		if (*N_dim == 0) {
			zlx complex_part = P_new / std::abs(P_new);
			*sum_norm += std::abs(P_new) * std::abs(P_new);
			ampl_arr[*multiindex] = std::abs(P_new);

			if (*multiindex == 0) {  // <-- first element of factor, *k_pi=0
				double phase = std::arg(complex_part) + (*k_pi) * M_PI * 2;

				*sum_1 += phase * (*multiindex);
				*sum_2 += (*multiindex) * (*multiindex);
				*sum_3 += phase;
				*sum_4 += (*multiindex);
				*phase_m1 = phase;
				phase_arr[*multiindex] = phase;

				(*multiindex)++;
			} else {
				// need to make phase like a straight line
				double phase = std::arg(complex_part) + (*k_pi) * M_PI * 2;

				if (phase - (*phase_m1) > (M_PI))  //<--- important!
				{
					(*k_pi)--;
					phase = phase - M_PI * 2;
				}
				if (phase - (*phase_m1) < (-M_PI))  //<--- important!
				{
					(*k_pi)++;
					phase = phase + M_PI * 2;
				}

				*sum_1 += phase * (*multiindex);
				*sum_2 += (*multiindex) * (*multiindex);
				*sum_3 += phase;
				*sum_4 += (*multiindex);
				*phase_m1 = phase;
				phase_arr[*multiindex] = phase;

				(*multiindex)++;
			}
		} else {
			(*N_dim)--;
			cycle_dim(dim, P_new, offset, r, multiindex, N_dim, k_pi, sum_norm, sum_1, sum_2, sum_3, sum_4, phase_m1, factors, ampl_arr, phase_arr);
		}
	}
	(*N_dim)++;
};

// replace factor with steering vector * polynomial (0,1, more) amplitude
void poly_steering(zlx * iappr, const int * iactr0, const int * splash, const int * splash_size, const int * rank, int ampl_order, bool extrap_flag, bool quant_flag, int num_levs, 
		double * out_coef1, double * out_coef2, bool save_flag, int * num_saved)
{
	/* Input parameters:
	   iappr - array with all factors
	   iactr0 - the first index of the factor that we are changing
	   splash - array with quantization sizes
	   splash_size - number of quantization sizes
	   rank 
	   ampl_order - order of polynomial to approximate ln of the amplitude (0 - steering vector, 1 - const*exp(i(ck+d)), 2 - exp(ak+b)*exp(i(ck+d)), where k -index);
	   if ampl_order>2 then QTT cannot be used
	   extrap_flag - true if extrapolation for tti
	   quant_flag - true if discretization of parameters
	   num_levs - number of levels for discretization of phases/phase decline 
	   out_coef1, out_coef2 - for common coefficients that are taken out (for discretization)
	   save_flag - if true, bad factors are approximated carefully
	   num_saved - array to count these bad factors
	*/
	int iactr, dimmlt;
	int n=1;
	zlx zone = 1.0;
	zlx zzero = 0.0;
	for(int d = 0; d < *splash_size; d++)
	{
		n *= splash[d];
	}
	zlx * factor = new zlx[*rank * n];
	int * offset = new int[*splash_size];
	int * splash_copy = new int[*splash_size];
	double * averages = new double[*rank];
	double * ampl_arr = new double[n];
	double * phase_arr = new double[n];
	double * ampl_coeff = new double[*rank * n];
	double * phase_coeff = new double[*rank * n]; // < don't use it currently
	zlx tempo, newtempo;
	double ftemp, ftemp2;

	// check if parameters are valid:
	if ((*splash_size>1)&&(ampl_order>2)) 
	{
		printf("Warning! Inconsistent parameters in poly_steering (*splash_size), changing ampl_order...\n");
		ampl_order=1;
	}
	if ((save_flag)&&(ampl_order>2)) 
	{
		printf("Warning! Inconsistent parameters in poly_steering (save_flag), changing ampl_order...\n");
		ampl_order=1;
	}
	if ((save_flag)&&(!(quant_flag))) 
	{
		printf("Warning! Inconsistent parameters in poly_steering (quant_flag), changing quant_flag...\n");
		quant_flag=true;
	}
	if ((save_flag)&&(extrap_flag)) 
	{
		printf("Warning! Inconsistent parameters in poly_steering, save_flag && extrap_flag, changing save_flag...\n");
		save_flag=false;
	}


	for(int r=0; r<*rank; r++) {averages[r]=0;}
	for(int r=0; r<*rank; r++) {out_coef1[r]=0;}
	for(int r=0; r<*rank; r++) {out_coef2[r]=0;}
	for(int r=0; r<2*(*rank); r++) {num_saved[r]=0;}

	// generate auxiliary arrays "offset", "factor"
	iactr=*iactr0;
	offset[0]=0;
	for(int d = 0; d < *splash_size; d++)
	{
		if (d>0) {offset[d]=offset[d-1]+splash[d-1]*(*rank);}
		splash_copy[d]=splash[d];
		for(int r = 0; r < *rank; r++)
		{
			for(int i = 0; i < splash[d]; i++)
			{
				factor[offset[d]+i+r*splash[d]]=iappr[iactr];
				iactr++;
			};
		};
	};

	// suppose that for each r=0,...,rank, phase_arr[k] is approximated by (coef[r]*k + out_coef2[r])
	double * coef = new double[*rank];
	for(int r = 0; r < *rank; r++)   // < -- each r is treated separately
	{
		zlx P;
		P=zone;
		int multiindex=0;
		int N_dim=*splash_size-1;
		int k_pi=0;
		double sum_norm=0;
		double sum_1=0;
		double sum_2=0;
		double sum_3=0;
		double sum_4=0;
		double phase_m1=0;

		// recursive function over quantized dimensions
		cycle_dim(splash_copy, P, offset, r, &multiindex, &N_dim, &k_pi, &sum_norm, &sum_1, &sum_2, &sum_3, &sum_4, &phase_m1, factor, ampl_arr, phase_arr);
		// output: sum_1, sum_2, sum_3, sum_4 (used to find coef[r], out_coef2[r]),
		// phase_arr, ampl_arr (phase and amplidude of factor: ampl_arr[k]*exp(i*phase_arr[k]), where k is the index of element)
		coef[r]=(sum_1 - sum_3*sum_4/(double) n)/(sum_2 - sum_4*sum_4/(double) n);    
		out_coef2[r]=(sum_3-coef[r]*sum_4)/(double) n;  // <-- free_coef

		// Uncomment to check the output:
		// for(int i=0; i<n; i++) {printf("Amplitude %d,  %f \n", r, ampl_arr[i]); }

		if (save_flag == true) {
			zlx scal_prod=0;
			zlx tempo, tempo2;
			double ampl_ave=0;
			double ampl_div=0;
			for(int i=0; i<n; i++) {ampl_ave=ampl_ave+ampl_arr[i];}
			for(int i=0; i<n; i++)
			{
				tempo.real(cos(coef[r] * i + out_coef2[r]));
				tempo.imag(sin(coef[r] * i + out_coef2[r]));
				tempo2.real(cos(phase_arr[i]));
				tempo2.imag(sin(phase_arr[i]));
				scal_prod += tempo*std::conj(tempo2);
				ampl_div += (ampl_arr[i]-ampl_ave/(double) n)*(ampl_arr[i]-ampl_ave/(double) n);
			}
			if ((std::abs(scal_prod)/(double) n) < PHASE_THRESHOLD) {
				num_saved[r]=1;  // use first R slots
				// Uncomment to check the output:
				for(int i=0; i<n; i++) {printf("Non-steering Phase %d, %f vs %f \n", r, coef[r] * i + out_coef2[r], phase_arr[i]); }
			}
			if (std::sqrt(ampl_div) > ampl_ave*AMPL_THRESHOLD) {
				num_saved[r+(*rank)]=1;  // use second R slots
				// Uncomment to check the output:
				for(int i=0; i<n; i++) {printf("Non-steering Amplitude %d, %f vs %f \n", r, ampl_ave/(double) n, ampl_arr[i]); }
				ampl_order=2;
			}
			averages[r]=ampl_ave/(double) n;
		}

		// Quantization: replace by discrete value from range [0,2pi]:
		if (quant_flag == true) {
			coef[r]=rounder(coef[r], num_levs);
		}

		if (ampl_order>0) 
		{
			for(int i = 0; i < n; i++) {
				ampl_coeff[i + r * n] = std::log(ampl_arr[i]);
			}
		}
	};

	// polynomial amplitude
	float * ord_coef = new float[ampl_order]; 
	double * vander_matrix = new double[ampl_order*n];

	if (ampl_order>0)
	{
		for(int i = 0; i < n; i++)
		{
			vander_matrix[i] = 1.0;
			for(int ord = 1; ord < ampl_order; ord++)
			{
				vander_matrix[i + ord * n] = vander_matrix[i + (ord-1) * n]*(i/(double) n);  // <-- Vandermonde matrix
			};
		};

		// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		int tiwork = 64 * n;
		double * twork = new double[tiwork];
		int tti_c;
		tti_c=n;
		char cN = 'N';
		int info = -153;
		dgels(&cN, &tti_c, &ampl_order, rank, vander_matrix, &tti_c, ampl_coeff, &tti_c, twork, &tiwork, &info); // <-- solve linear system with Vandermonde matrix
		// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		iactr=*iactr0;
		for(int r = 0; r < *rank; r++)
		{
			for(int ord = 0; ord < ampl_order; ord++)
			{
				if (ord>0) {
					if (quant_flag) {
						ord_coef[ord] = rounder_amp(ampl_coeff[ord+r*n]);
					} else {
						ord_coef[ord] = ampl_coeff[ord+r*n];  // <-- polynomial coefficients to approximate the amplitude
					}
				} else {
					ord_coef[ord] = ampl_coeff[ord+r*n];  // <-- polynomial coefficients to approximate the amplitude
				}
			};

			// write down the result:
			if ((*splash_size==1)&&(save_flag==false))
			{
				// save extrapolated factor <--- no quantization!
				// here ampl_order can be any positive integer
				for(int i = 0; i < n; i++)
				{
					ftemp =0;
					ftemp2 =1;
					for(int ord = 0; ord < ampl_order; ord++)
					{
						ftemp+=ord_coef[ord] * ftemp2;
						if (extrap_flag==true) {
							ftemp2 *= ((i+n)/(double) n);
						} else {
							ftemp2 *= (i/(double) n);
						}
					};
					if (ampl_order>0) {
						iappr[iactr]=std::exp(ftemp-ord_coef[0]); // <-- take out constant
					} else {
						iappr[iactr]=zone;
					}
					out_coef1[r] = ord_coef[0]; // < -- save it into output coefficient

					if (extrap_flag==true) {
						tempo.real(cos(coef[r] * (i+n)));
						tempo.imag(sin(coef[r] * (i+n)));
					} else {
						tempo.real(cos(coef[r] * i));
						tempo.imag(sin(coef[r] * i));
					}
					iappr[iactr]=iappr[iactr]*tempo;					
					// end approximate/extrapolate phase
					// obtained factor is saved into iappr
					iactr++;
				};
			};
		}
		// In case of quantization/saving factors:
		if ((*splash_size>1)||((*splash_size==1)&&(save_flag==true)))  // note that here ampl_order<=2, it is used explicitly
		{
			iactr=*iactr0;
			dimmlt = 1.0;

			for(int r = 0; r < *rank; r++) {
				for(int ord = 0; ord < ampl_order; ord++) {
					if (ord>0) {
						if (quant_flag) {
							ord_coef[ord] = rounder_amp(ampl_coeff[ord+r*n]);
						} else {
							ord_coef[ord] = ampl_coeff[ord+r*n];  // <-- polynomial coefficients to approximate the amplitude
						}
					} else {
						ord_coef[ord] = ampl_coeff[ord+r*n];  // <-- polynomial coefficients to approximate the amplitude
					}
				}

				if ((save_flag)&&(num_saved[r+(*rank)]==0)) {
					out_coef1[r]=std::log(averages[r]);
				} else {
					out_coef1[r]=ord_coef[0];
				}

				if (extrap_flag==true)
				{
					out_coef2[r]=out_coef2[r]+coef[r] * n;
					if (ampl_order==2) {
						out_coef1[r]=out_coef1[r]+ord_coef[1];
					}
				}
			}
			zlx * complex_coef = new zlx[*rank];
			for(int r=0; r<*rank; r++) {complex_coef[r]=zone;}
			for(int d = 0; d < *splash_size; d++)  // cycle over quantized dimensions
			{
				for(int r = 0; r < *rank; r++)
				{
					if ((save_flag == false)||(num_saved[r] == 0)) {
						for(int i = 0; i < splash[d]; i++)
						{
							iappr[iactr].real(cos(coef[r] * i * dimmlt));
							iappr[iactr].imag(sin(coef[r] * i * dimmlt));
							if (ampl_order==2)
							{
								if ((save_flag==false)||(num_saved[r+(*rank)]==1)) {
									if (quant_flag) {
										iappr[iactr]=iappr[iactr]*std::exp(rounder_amp(ampl_coeff[1+r*n]) * i * dimmlt/(double) n);
									} else {
										iappr[iactr]=iappr[iactr]*std::exp(ampl_coeff[1+r*n] * i * dimmlt/(double) n);
									}
								}
							}
							iactr++;
							// obtained factor is saved into iappr
						};
					} else {
						zlx first_el=zone;
						if (std::abs(iappr[iactr])>0.0001) {   // filter bad cases
							first_el=iappr[iactr];
						} 
						complex_coef[r]=complex_coef[r]*first_el;
						for(int i = 0; i < splash[d]; i++)
						{
							iappr[iactr]=iappr[iactr]/first_el;
							double phase = rounder(std::arg(iappr[iactr]), num_levs);
							iappr[iactr].real(cos(phase));
							iappr[iactr].imag(sin(phase));
							if (ampl_order==2)
							{
								if (num_saved[r+(*rank)]==1) {
									iappr[iactr]=iappr[iactr]*std::exp(rounder_amp(ampl_coeff[1+r*n]) * i * dimmlt/(double) n);
								}
							}
							iactr++;
						};

					}
				};
				dimmlt *= splash[d];
			};
			for(int r=0; r<*rank; r++) 
			{
				if ((save_flag == true)&&(num_saved[r] == 1)) {
					out_coef1[r]=std::log(averages[r]);
					out_coef2[r]=std::arg(complex_coef[r]);
				}
			}
			delete[] complex_coef;
		}
		delete[] twork;
	}	
	else
	{
		// write down the result (steering vector)
		iactr=*iactr0;
		dimmlt = 1.0;
		for(int d = 0; d < *splash_size; d++)
		{
			for(int r = 0; r < *rank; r++)
			{
				for(int i = 0; i < splash[d]; i++)
				{
					iappr[iactr].real(cos(coef[r] * i * dimmlt));
					iappr[iactr].imag(sin(coef[r] * i * dimmlt));
					iactr++;
				};
			};
			dimmlt *= splash[d];
		};
	};
	delete[] coef;
	delete[] factor;
	delete[] offset;
	delete[] splash_copy;
	delete[] averages;
	delete[] ampl_arr;
	delete[] phase_arr;
	delete[] ampl_coeff;
	delete[] phase_coeff;

	// polynomial amplitude
	delete[] ord_coef;
	delete[] vander_matrix;
}

void QuantizedFactorGnuprint(const char * refname, const std::vector<uint64_t> & sizes, int rank, const zlx * fmtx)
{
	char gnuname[240];
	char picname[240];
	char datname[240];

	FILE * fddat;
	FILE * fdgnu;

	int szmult;
	int szline;

	szmult = 1;
	szline = 0;
	for(int i = 0; i < sizes.size(); i++)
	{
		szmult *= sizes[i];
		szline += sizes[i];
	};

	sprintf(gnuname, "columbia/%s.gnu", refname);
	sprintf(picname, "columbia/%s.png", refname);

	fdgnu = fopen(gnuname, "w");
	fprintf(fdgnu, "set terminal pngcairo size 1080, 1080\nset output '%s'\nset tics font ',20'\nset xlabel 'Real part' font ',20'\nset ylabel 'Imag part' font ',20'\nset title 'Factor values' font ',24'\n", picname);
	
	fprintf(fdgnu, "plot ");

	zlx * onerank = new zlx[szline * rank];
	
	int shift = 0;
	for(int d = 0; d < sizes.size(); d++)
	{
		for(int r = 0; r < rank; r++)
		{
			for(int i = 0; i < sizes[d]; i++)
			{
				onerank[r * szline + shift + i] = fmtx[shift * rank + r * sizes[d] + i];
			};
		};
		shift += sizes[d];		
	};

	for(int r = 0; r < rank; r++)
	{
		CanonicalTensor<zlx> v(sizes, 1, onerank + r * szline);
		
		sprintf(datname, "columbia/%s_r%d.txt", refname, r);
		fddat = fopen(datname, "w");

		for(int i = 0; i < szmult; i++)
		{
			zlx el = v.get_element(i);
			fprintf(fddat, "%e %e\n", el.real(), el.imag());
		};

		fclose(fddat);

		if (r != 0) fprintf(fdgnu, ", ");
		fprintf(fdgnu, "'%s' title 'r %d' with linespoints lw 3 pt 7 ps 2", datname, r + 1);
	};
	delete[] onerank;

	fprintf(fdgnu, "\nreset\n");
	fclose(fdgnu);
};

void QuantizedFactorFullScalePrint(const char * refname, const std::vector<uint64_t> & rx_sizes, const std::vector<uint64_t> & tx_sizes_1, const std::vector<uint64_t> & tx_sizes_2, const std::vector<uint64_t> & sc_sizes, const std::vector<uint64_t> & tti_sizes, int rank, zlx * iappr)
{
	char faname[240];
	int shift;

	int rx_line = 0;
	int tx1_line = 0;
	int tx2_line = 0;
	int sc_line = 0;
	int tti_line = 0;

	for(int i = 0; i < rx_sizes.size(); i++) rx_line += rx_sizes[i];
	for(int i = 0; i < tx_sizes_1.size(); i++) tx1_line += tx_sizes_1[i];
	for(int i = 0; i < tx_sizes_2.size(); i++) tx2_line += tx_sizes_2[i];
	for(int i = 0; i < sc_sizes.size(); i++) sc_line += sc_sizes[i];
	for(int i = 0; i < tti_sizes.size(); i++) tti_line += tti_sizes[i];

	std::vector<uint64_t> full_sizes;

	for(int i = 0; i < rx_sizes.size(); i++) full_sizes.push_back(rx_sizes[i]);
	for(int i = 0; i < tx_sizes_1.size(); i++) full_sizes.push_back(tx_sizes_1[i]);
	for(int i = 0; i < tx_sizes_2.size(); i++) full_sizes.push_back(tx_sizes_2[i]);
	for(int i = 0; i < sc_sizes.size(); i++) full_sizes.push_back(sc_sizes[i]);
	for(int i = 0; i < tti_sizes.size(); i++) full_sizes.push_back(tti_sizes[i]);

	//scaling1
	shift = 0;

	double * lognorms = new double[rank];

	for(int r = 0; r < rank; r++) lognorms[r] = 0.0;

	for(int d = 0; d < full_sizes.size(); d++)
	{
		for(int r = 0; r < rank; r++)
		{
			double nrm = 0.0;
			for(int i = 0; i < full_sizes[d]; i++)
			{
				nrm += std::norm(iappr[i + shift]);
			};
			shift += full_sizes[d];

			lognorms[r] += log(nrm) / (2.0 * full_sizes.size());
		};
	};

	shift = 0;
	for(int d = 0; d < full_sizes.size(); d++)
	{
		for(int r = 0; r < rank; r++)
		{
			double nrm = 0.0;
			for(int i = 0; i < full_sizes[d]; i++)
			{
				nrm += std::norm(iappr[i + shift]);
			};
			nrm = sqrt(nrm);
			
			for(int i = 0; i < full_sizes[d]; i++)
			{
				iappr[i + shift] *= exp(lognorms[r]) / nrm;
			};

			shift += full_sizes[d];
		};
	};

	delete[] lognorms;

	//printing
	shift = 0;

	sprintf(faname, "%s_rx", refname);
	QuantizedFactorGnuprint(faname, rx_sizes, rank, iappr + shift);
	shift += rx_line * rank;

	sprintf(faname, "%s_tx1", refname);
	QuantizedFactorGnuprint(faname, tx_sizes_1, rank, iappr + shift);
	shift += tx1_line * rank;

	sprintf(faname, "%s_tx2", refname);
	QuantizedFactorGnuprint(faname, tx_sizes_2, rank, iappr + shift);
	shift += tx2_line * rank;

	sprintf(faname, "%s_sc", refname);
	QuantizedFactorGnuprint(faname, sc_sizes, rank, iappr + shift);
	shift += sc_line * rank;

	sprintf(faname, "%s_tti", refname);
	QuantizedFactorGnuprint(faname, tti_sizes, rank, iappr + shift);
	shift += tti_line * rank;
};

std::vector<uint64_t> splash235(int num) {
	std::vector<uint64_t> result;
	result.resize(0);

	while ((num / 2) * 2 == num) {
		num /= 2;
		result.push_back(2);
	};

	while ((num / 3) * 3 == num) {
		num /= 3;
		result.push_back(3);
	};

	while ((num / 5) * 5 == num) {
		num /= 5;
		result.push_back(5);
	};

	if (num != 1) printf("QTT size splashing into 2,3,5 multipliers failed.\n");

	return result;
};

void grass_cosine_row(int m, int n, const zlx* m1, const zlx* m2, double* out) {
	zlx* m1c = new zlx[n * m];
	zlx* m2c = new zlx[n * m];

	zlx* t1 = new zlx[n];
	zlx* t2 = new zlx[n];

	zlx* core = new zlx[m * m];

	int lwork = 64 * n;
	int info = 0;

	zlx* zwork = new zlx[lwork];
	double* dwork = new double[lwork];

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			m1c[j + i * n] = std::conj(m1[i + m * j]);
			m2c[j + i * n] = std::conj(m2[i + m * j]);
		};
	};

	zgeqrf(&n, &m, m1c, &n, t1, zwork, &lwork, &info);
	zgeqrf(&n, &m, m2c, &n, t2, zwork, &lwork, &info);

	zungqr(&n, &m, &m, m1c, &n, t1, zwork, &lwork, &info);
	zungqr(&n, &m, &m, m2c, &n, t2, zwork, &lwork, &info);

	char cC = 'C';
	char cN = 'N';

	zlx zzero = 0.0;
	zlx zone = 1.0;

	zgemm(&cC, &cN, &m, &m, &n, &zone, m1c, &n, m2c, &n, &zzero, core, &m);
	zgesvd(&cN, &cN, &m, &m, core, &m, out, NULL, &m, NULL, &m, zwork, &lwork, dwork, &info);

	delete[] m1c;
	delete[] m2c;

	delete[] zwork;
	delete[] dwork;

	delete[] t1;
	delete[] t2;

	delete[] core;
};

// Artemasov style begin

std::complex<double>* simulate(std::complex<double> ****MyChannel, int ue_idx, int num_threads, int n_rx, int n_tti, int n_sc, int n_tx, int quantize_rx, 
        int quantize_tx, int quantize_sc, int quantize_tti, int tti_order, int sc_format, int tx_format, int rx_format, int rank, int tti_cut, int n_tx_onedim_1, 
        int n_tx_onedim_2, int firstALS_iterations, int tailALS_iterations, int extfft_multiplier, double extfft_resolution, double SNR){
/*
std::complex<double>**** simulate(std::complex<double> ****MyChannel, int ue_idx, int num_threads, int n_rx, int n_tti, int n_sc, int n_tx, int quantize_rx, 
        int quantize_tx, int quantize_sc, int quantize_tti, int tti_order, int sc_format, int tx_format, int rx_format, int rank, int tti_cut, int n_tx_onedim_1, 
        int n_tx_onedim_2, int firstALS_iterations, int tailALS_iterations, int extfft_multiplier, double extfft_resolution, double SNR){
*/
    omp_set_num_threads(num_threads);

/*
int main(int argc, char** argv) {
	int n_rx = 4;
	int n_tx = 32;  
	int n_sc = 48;    // 96
	int n_tti = 96;  // 200, 128

	// Enable or disable quantization by factors tx, sc and tti
	int quantize_rx = 1;
	int quantize_tx = 1;
	int quantize_sc = 1;
	int quantize_tti = 1;

	int tti_order = 2;  // order of extrapolation, order>=0
	int sc_format = 1;  // sc representation format, 0 - ALS-based, 1 - find phase
	int tx_format = 0;  // tx representation format, 0 - ALS-based, 1 - find phase
	int rx_format = 0;  // rx representation format, 0 - ALS-based, 1 - find phase
	int tx_order = 1;   // tx amplitude representetion order

	int rank = 8;

	int tti_cut = 16;
	int n_tx_onedim_1 = 8;
	// int n_tx_onedim_1 = 2;
	int n_tx_onedim_2 = 2;

	// Enable or disable accurate approximation of bad (non-steering) vectors (factors)
	bool save_flag = false;

	int firstALS_iterations = 50;
	int tailALS_iterations = 10;

	int extfft_multiplier = 30; // 30
	double extfft_resolution = 1.0;
*/
    
    int magic_size = 528;
    std::complex<double> *results_tensor  = new std::complex<double>[2*magic_size+1];

    // My addition
    int ue=ue_idx;

  	// Enable or disable accurate approximation of bad (non-steering) vectors (factors)
	bool save_flag = false;
	int tx_order = 1;   // tx amplitude representetion order

//	double SNR = 30.0;  // 30.0
	double noise = pow(10.0, -SNR / 10.0) / n_sc;

    
    char filename[240];
	char gnuname[240];
	char picname[240];
	char refname[240];

	char cO = 'O';
	char cN = 'N';
	char cC = 'C';

	int info = -153;
	zlx zone = 1.0;
	zlx zzero = 0.0;
	int izero = 0;
	int ione = 1;
	double dzero = 0.0;

	int n_tx_twodim = n_tx / n_tx_onedim_1 / n_tx_onedim_2;
	int tti_steps = n_tti / tti_cut;

	zlx* all_t = new zlx[n_rx * n_tx * n_sc * n_tti];
	zlx* all_h = new zlx[n_rx * n_tx * n_sc * n_tti];
	zlx* all_i = new zlx[n_rx * n_tx * n_sc * n_tti];
	zlx* all_e = new zlx[n_rx * n_tx * n_sc * n_tti];
/*
	for (int rx = 0; rx < n_rx; rx++) {
		char fname[240];

		//path to read data
		//sprintf(fname, "3gpp3_ue0_rx%d_data.txt", rx);
		//sprintf(fname, "3gpp3_sum2_ue0_rx%d_data.txt", rx);
		//sprintf(fname, "berlin3_ue0_rx%d_data.txt", rx);
		sprintf(fname, "new32_ue0_rx%d_data.dat", rx);
		//sprintf(fname, "short32_ue0_rx%d_data.txt", rx);
		//sprintf(fname, "debug8_ue0_rx%d_data.txt", rx);
		//sprintf(fname, "new45_ue1_rx%d_data.dat", rx);
		//sprintf(fname, "new128_sum2_ue0_rx%d_data.txt", rx);
		//sprintf(fname, "plastic32-6-60_rx%d_data.txt", rx);
		//sprintf(fname, "unsummed32_ue0_rx%d_data.dat", rx);
		// sprintf(fname, "unsummed128_ue0_rx%d_data.txt", rx);

		FILE* fd = fopen(fname, "r");

		int dummy_tx, dummy_sc, dummy_tti;

		fscanf(fd, "%d %d %d\n", &dummy_tx, &dummy_sc, &dummy_tti);
		printf("Loading %dx%dx%d tensor...", dummy_tx, dummy_sc, dummy_tti);
		fflush(stdout);

		for (int tti = 0; tti < n_tti; tti++) {
			for (int sc = 0; sc < n_sc; sc++) {
				for (int tx = 0; tx < n_tx; tx++) {
					double re, im;
					fscanf(fd, "(%lf %lf) \n", &re, &im);

					all_t[rx + n_rx * (tx + n_tx * (sc + n_sc * tti))].real(re);
					all_t[rx + n_rx * (tx + n_tx * (sc + n_sc * tti))].imag(im);
				};
				fscanf(fd, "\n");
			};
			fscanf(fd, "\n");
		};
		fclose(fd);

		printf("done.\n");
		fflush(stdout);
	};
*/
    /*
    double summ1 = 0;
    double summ2 = 0;

    for (int rx = 0; rx < n_rx; rx++) {
        for (int tti = 0; tti < n_tti; tti++) {
            for (int sc = 0; sc < n_sc; sc++) {
                for (int tx = 0; tx < n_tx; tx++) {
                    if(tx<128){
                        summ1+=norm(MyChannel[rx][tti][sc][tx]);
                    }else{
                        summ2+=norm(MyChannel[rx][tti][sc][tx]);
                    }
                };
            };
        };
    };
    summ1 = sqrt(summ1);
    summ2 = sqrt(summ2);

    std::cout<<"summ1= "<<1/summ1<<" summ2= "<<1/summ2<<std::endl;
*/
    for (int rx = 0; rx < n_rx; rx++) {
        for (int tti = 0; tti < n_tti; tti++) {
            for (int sc = 0; sc < n_sc; sc++) {
                for (int tx = 0; tx < n_tx; tx++) {
                    if(tx<256){
                        all_t[rx + n_rx * (tx + n_tx * (sc + n_sc * tti))] = MyChannel[rx][tti][sc][tx];//summ1;
                    }else{
                        all_t[rx + n_rx * (tx + n_tx * (sc + n_sc * tti))] = MyChannel[rx][tti][sc][tx];//summ2;
                    }
                };
            };
        };
    };
    // Artemasov style complete


    // My code scal for saving
    zlx *scalforsaving = new zlx[n_tti];

	// scaling
	for (int tti = 0; tti < n_tti; tti++) {
		int sc_size = n_tx * n_rx * n_sc;
		int ione = 1;

        zlx scal = dznrm2(&sc_size, all_t + sc_size * tti, &ione);
//        zlx summ_my=0;
//        for(int ii=0;ii<sc_size;ii++){
//            summ_my += norm( all_t[ii+sc_size * tti] );
//            if((ii<50)){
//                std::cout<<tti<<"  "<<ii<<"  "<<summ_my<<"  "<<all_t[ii+sc_size * tti]<<std::endl;
//            }
//        }

        scalforsaving[tti] = scal;
        std::cout<<sc_size<<"  "<<tti<<"  "<<scal<<std::endl;

        scal = 1. / scal;
        zscal(&sc_size, &scal, all_t + sc_size * tti, &ione);

//		zlx scal = 1.0 / dznrm2(&sc_size, all_t + sc_size * tti, &ione);
//		zscal(&sc_size, &scal, all_t + sc_size * tti, &ione);
	};

	int ext_tx_1 = n_tx_onedim_1 * extfft_multiplier;
	//int ext_tx_2 = n_tx_onedim_2 * extfft_multiplier;
	int ext_sc = n_sc * extfft_multiplier;
	int ext_szfull = ext_tx_1 * ext_sc;

	zlx* extfft = new zlx[ext_szfull];
	double* extavg = new double[ext_szfull];
	FFTvalue* fftdata = new FFTvalue[ext_szfull];

	DFTI_DESCRIPTOR_HANDLE fft_desc;
	MKL_LONG num_dim = 2;

	MKL_LONG sizes[2];
	sizes[0] = ext_sc;
	sizes[1] = ext_tx_1;
	//sizes[2] = ext_tx_1;

	DftiCreateDescriptor(&fft_desc, DFTI_DOUBLE, DFTI_COMPLEX, num_dim, sizes);
	DftiCommitDescriptor(fft_desc);

	dcopy(&ext_szfull, &dzero, &izero, extavg, &ione);

	for (int rx = 0; rx < n_rx; rx++) {
		for (int tx2 = 0; tx2 < n_tx_onedim_2; tx2++) {
			for (int tx_two = 0; tx_two < n_tx_twodim; tx_two++) {
				for (int tti = 0; tti < tti_cut; tti++) {
					zcopy(&ext_szfull, &zzero, &izero, extfft, &ione);

					for (int tx1 = 0; tx1 < n_tx_onedim_1; tx1++) {
						for (int sc = 0; sc < n_sc; sc++) {
							extfft[tx1 + ext_tx_1 * sc] = all_t[rx + n_rx * (tx1 + n_tx_onedim_1 * (tx2 + n_tx_onedim_2 * (tx_two + n_tx_twodim * (sc + n_sc * tti))))];
						}
					}

					DftiComputeForward(fft_desc, extfft);
					zlx scal = 1.0 / sqrt(ext_szfull);
					zscal(&ext_szfull, &scal, extfft, &ione);

					for (int tx1 = 0; tx1 < ext_tx_1; tx1++) {
						for (int sc = 0; sc < ext_sc; sc++) {
							extavg[tx1 + ext_tx_1 * sc] += std::norm(extfft[tx1 + ext_tx_1 * sc]);
						}
					}
				}
			}
		}
	}

	sprintf(filename, "columbia/extfft_averaged.txt");
	sprintf(gnuname, "columbia/extfft_averaged.gnu");
	sprintf(picname, "columbia/extfft_averaged.png");		

	FILE * fd = fopen(filename, "w");
	FILE * fdgnu = fopen(gnuname, "w");

	for (int tx1 = 0; tx1 < ext_tx_1; tx1++) {
		for (int sc = 0; sc < ext_sc; sc++) {

			fprintf(fd, "%e %e %e\n", tx1 * 2.0 * M_PI / (double) ext_tx_1, sc * 2.0 * M_PI / (double) ext_sc, sqrt(extavg[tx1 + ext_tx_1 * sc]));

			fftdata[tx1 + ext_tx_1 * sc].tx_phase_1 = tx1 * 2.0 * M_PI / (double)ext_tx_1;
			fftdata[tx1 + ext_tx_1 * sc].sc_phase = sc * 2.0 * M_PI / (double)ext_sc;
			fftdata[tx1 + ext_tx_1 * sc].value = sqrt(extavg[tx1 + ext_tx_1 * sc]);
		}
		fprintf(fd, "\n");
	}

	fprintf(fdgnu, "set terminal pngcairo size 1920, 1080\nset output '%s'\nset pm3d map\nset nokey\nset tics font ',20'\nset xlabel 'SC phase' font ',20'\nset ylabel 'TX phase' font ',20'\nset title 'FFT, row norms over tti axis' font ',24'\n", picname);
	fclose(fd);

	qsort(fftdata, ext_szfull, sizeof(FFTvalue), FFTCompare);
	DftiFreeDescriptor(&fft_desc);
	delete[] extfft;
	delete[] extavg;

	int fftctr = 0;
	int peakctr = 0;

	double* tx_phases_1 = new double[rank];
	//double* tx_phases_2 = new double[rank];
	double* sc_phases = new double[rank];

	// careful peak handling
	while (peakctr < rank) {
		bool notclose = true;

		// checking if a current point is close to an already selected peak
		for (int r = 0; r < peakctr; r++) {
			double tx_phasediff_1 = fftdata[fftctr].tx_phase_1 - tx_phases_1[r];
			//double tx_phasediff_2 = fftdata[fftctr].tx_phase_2 - tx_phases_2[r];
			double sc_phasediff = fftdata[fftctr].sc_phase - sc_phases[r];

			while (tx_phasediff_1 > M_PI) {
				tx_phasediff_1 -= 2.0 * M_PI;
			}

			while (tx_phasediff_1 < -M_PI) {
				tx_phasediff_1 += 2.0 * M_PI;
			}

			/*while (tx_phasediff_2 > M_PI) {
			  tx_phasediff_2 -= 2.0 * M_PI;
			  }

			  while (tx_phasediff_2 < -M_PI) {
			  tx_phasediff_2 += 2.0 * M_PI;
			  }*/

			while (sc_phasediff > M_PI) {
				sc_phasediff -= 2.0 * M_PI;
			};

			while (sc_phasediff < -M_PI) {
				sc_phasediff += 2.0 * M_PI;
			};

			tx_phasediff_1 = std::abs(tx_phasediff_1);
			//tx_phasediff_2 = std::abs(tx_phasediff_2);
			sc_phasediff = std::abs(sc_phasediff);

			if (
					tx_phasediff_1 < 2.0 * M_PI * extfft_resolution / (double)n_tx_onedim_1 &&
					//                    tx_phasediff_2 < 2.0 * M_PI * extfft_resolution / (double)n_tx_onedim_2 &&
					sc_phasediff < 2.0 * M_PI * extfft_resolution / (double)n_sc) {
				notclose = false;
			}
		};

		if (notclose) {
			tx_phases_1[peakctr] = fftdata[fftctr].tx_phase_1;
			//            tx_phases_2[peakctr] = fftdata[fftctr].tx_phase_2;
			sc_phases[peakctr] = fftdata[fftctr].sc_phase;

			fprintf(fdgnu, "set label %d at %e,%e,%e front 'Peak #%d' tc 'green' font ',25' point pointtype 2 pointsize 4.5\n", peakctr + 1,  sc_phases[peakctr], tx_phases_1[peakctr], 0.0, peakctr + 1);		
			printf("Selected a peak %d at TX1 phase %f and SC phase %f (value %e).\n", peakctr, tx_phases_1[peakctr], sc_phases[peakctr], fftdata[fftctr].value);
			peakctr++;
		};

		fftctr++;
	};

	fprintf(fdgnu, "splot '%s' u 2:1:3 w pm3d\nreset\n", filename);
	fclose(fdgnu);

	delete[] fftdata;

	std::vector<uint64_t> rx_splash = splash235(n_rx);
	std::vector<uint64_t> tx_splash_1 = splash235(n_tx_onedim_1);
	std::vector<uint64_t> tx_splash_2 = splash235(n_tx_onedim_2);
	std::vector<uint64_t> sc_splash = splash235(n_sc);
	std::vector<uint64_t> tti_splash = splash235(tti_cut);

	uint64_t rx_dim = rx_splash.size();
	uint64_t tx_dim_1 = tx_splash_1.size();
	uint64_t tx_dim_2 = tx_splash_2.size();
	uint64_t sc_dim = sc_splash.size();
	uint64_t tti_dim = tti_splash.size();

	std::vector<uint64_t> parafac_sizes(0);

	int num_rx;
	int num_tx_1;
	int num_tx_2;
	int num_sc;
	int num_tti;

	int num_saved_tx1_total = 0;
	int num_saved_tx2_total = 0;
	int num_saved_sc_total = 0;	
	int num_saved_amplitudes_total = 0;

	std::vector<uint64_t> rx_sizes;
	std::vector<uint64_t> tx_sizes_1;
	std::vector<uint64_t> tx_sizes_2;
	std::vector<uint64_t> sc_sizes;
	std::vector<uint64_t> tti_sizes;

	if (quantize_rx) {
		num_rx=0;
		for (uint64_t rx_d = 0; rx_d < rx_dim; rx_d++) {
			parafac_sizes.push_back(rx_splash[rx_d]);
			rx_sizes.push_back(rx_splash[rx_d]);
			num_rx = num_rx + (int)rx_splash[rx_d]-1; 
		};
	} else {
		parafac_sizes.push_back(n_rx);
		rx_sizes.push_back(n_rx);
		num_rx=n_rx-1;
	}

	if (quantize_tx) {
		num_tx_1=0;
		for (uint64_t tx_d_1 = 0; tx_d_1 < tx_dim_1; tx_d_1++) {
			parafac_sizes.push_back(tx_splash_1[tx_d_1]);
			tx_sizes_1.push_back(tx_splash_1[tx_d_1]);
			num_tx_1 = num_tx_1 + (int)tx_splash_1[tx_d_1]-1; 
		}

		num_tx_2=0;
		for (uint64_t tx_d_2 = 0; tx_d_2 < tx_dim_2; tx_d_2++) {
			parafac_sizes.push_back(tx_splash_2[tx_d_2]);
			tx_sizes_2.push_back(tx_splash_2[tx_d_2]);
			num_tx_2 = num_tx_2 + (int)tx_splash_2[tx_d_2]-1; 
		}
	} else {
		parafac_sizes.push_back(n_tx_onedim_1);
		tx_sizes_1.push_back(n_tx_onedim_1);
		num_tx_1 = n_tx_onedim_1 - 1;

		parafac_sizes.push_back(n_tx_onedim_2);
		tx_sizes_2.push_back(n_tx_onedim_2);
		num_tx_2 = n_tx_onedim_2 - 1;
	}

	if (quantize_sc) {
		num_sc=0;
		for (uint64_t sc_d = 0; sc_d < sc_dim; sc_d++) {
			parafac_sizes.push_back(sc_splash[sc_d]);
			sc_sizes.push_back(sc_splash[sc_d]);
			num_sc = num_sc + (int)sc_splash[sc_d]-1; 
		};
	} else {
		parafac_sizes.push_back(n_sc);
		sc_sizes.push_back(n_sc);
		num_sc=n_sc - 1;
	}

	if (quantize_tti) {
		num_tti=0;
		for (uint64_t tti_d = 0; tti_d < tti_dim; tti_d++) {
			parafac_sizes.push_back(tti_splash[tti_d]);
			tti_sizes.push_back(tti_splash[tti_d]);
			num_tti = num_tti + (int)tti_splash[tti_d]-1; 
		};
	} else {
		parafac_sizes.push_back(tti_cut);
		tti_sizes.push_back(tti_cut);
		num_tti=tti_cut-1;
	}

	int lsm = n_tx_onedim_1 * n_sc;
	int lsn = rank;
	int lsrs = tti_cut * n_rx * n_tx_onedim_2;

	zlx* lsmatrix = new zlx[lsm * lsn];
	zlx* lscoeff = new zlx[lsm * lsrs];

	zlx* tmp_rx_tx2_tti = new zlx[lsrs];
	zlx* tmp_rx_tx2_tti_factors = new zlx[n_rx + n_tx_onedim_2 + tti_cut];

	zlx* tmp_rx = new zlx[rank * n_rx];
	zlx* tmp_tx2 = new zlx[rank * n_tx_onedim_2];
	zlx* tmp_tti = new zlx[rank * tti_cut];

	int lwork = 64 * lsm * lsn + 64 * lsrs;
	zlx* zwork = new zlx[lwork];
	double* dwork = new double[lwork];

	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<double> d;

	std::vector<zlx> tensor_v(n_rx * tti_cut * n_sc * n_tx_onedim_1 * n_tx_onedim_2);
	std::vector <int> parafac_sizes_long = {n_rx, n_tx_onedim_1, n_tx_onedim_2, n_sc, tti_cut};
	zlx* iappr_long = new zlx[rank * (n_rx + n_tx_onedim_1 + n_tx_onedim_2 + n_sc + tti_cut)];
	zlx* iappr = new zlx[rank * (n_rx + n_tx_onedim_1 + n_tx_onedim_2 + n_sc + tti_cut)];
	std::vector<zlx> ttifactor_full(tti_cut*rank);

	std::vector<zlx> rxfactor_v(n_rx);
	std::vector<zlx> txfactor_v_1(n_tx_onedim_1);
	std::vector<zlx> txfactor_v_2(n_tx_onedim_2);
	std::vector<zlx> scfactor_v(n_sc);
	std::vector<zlx> ttifactor_v(tti_cut);

	int iactr;
	int iactr_rx, iactr_tx_1, iactr_tx_2, iactr_sc, iactr_tti;
	double global_sum_diff = 0.0, global_sum_true = 0.0; 

    // ************************************* Main cycle ********************************************************
	for (int tx_two = 0; tx_two < n_tx_twodim; tx_two++) {
		printf("\n\nWorking with tx second dim %d...\n", tx_two);

		// initial approximation -- time, matrix
		for (int tx_1 = 0; tx_1 < n_tx_onedim_1; tx_1++) {
			for (int sc = 0; sc < n_sc; sc++) {
				for (int r = 0; r < rank; r++) {
					zlx ctx_1, csc;

					ctx_1.real(cos(tx_1 * tx_phases_1[r]));
					ctx_1.imag(sin(tx_1 * tx_phases_1[r]));

					csc.real(cos(sc * sc_phases[r]));
					csc.imag(sin(sc * sc_phases[r]));

					lsmatrix[tx_1 + n_tx_onedim_1 * (sc + n_sc * r)] = ctx_1 * csc;
				}
			}
		}

		// initial approximation -- time, right side
		for (int rx = 0; rx < n_rx; rx++) {
			for (int tx_1 = 0; tx_1 < n_tx_onedim_1; tx_1++) {
				for (int tx_2 = 0; tx_2 < n_tx_onedim_2; tx_2++) {
					for (int sc = 0; sc < n_sc; sc++) {
						for (int tti = 0; tti < tti_cut; tti++) {
							lscoeff[tx_1 + n_tx_onedim_1 * (sc + n_sc * (rx + n_rx * (tx_2 + n_tx_onedim_2 * tti)))] = 
								all_t[rx + n_rx * (tx_1 + n_tx_onedim_1 *(tx_2 + n_tx_onedim_2 * (tx_two + n_tx_twodim * (sc + n_sc * tti))))];
						}
					}
				}
			}
		}

		int size_temp = lsm * lsrs;
		double rsnrm = dznrm2(&size_temp, lscoeff, &ione);

		printf("Norm: %lf\n", rsnrm);
		// initial approximation -- time, least squares
		zgels(&cN, &lsm, &lsn, &lsrs, lsmatrix, &lsm, lscoeff, &lsm, zwork, &lwork, &info);

		double rsres = 0.0;
		int size_tail = lsm - lsn;
		for (int j = 0; j < lsrs; j++) {
			double rsrespart = dznrm2(&size_tail, lscoeff + lsn + j * lsm, &ione);
			rsres += rsrespart * rsrespart;
		};
		rsres = sqrt(rsres);

		printf("Initial FFT approximation error (tx1 X sc X unstructured): %.3f.\n", rsres / rsnrm);

		int ptr_rx = 0;
		int ptr_tx2 = 0;
		int ptr_tti = 0;
		for (int r = 0; r < rank; r++) {
			for (int i = 0; i < lsrs; i++) {
				tmp_rx_tx2_tti[i] = lscoeff[r + i * lsm];
			}

			std::vector <uint64_t> curr_sizes;

			for (uint64_t rx_d = 0; rx_d < rx_sizes.size(); rx_d++) {
				curr_sizes.push_back(rx_sizes[rx_d]);
			}

			for (uint64_t tx_d_2 = 0; tx_d_2 < tx_sizes_2.size(); tx_d_2++) {
				curr_sizes.push_back(tx_sizes_2[tx_d_2]);
			}

			for (uint64_t tti_d = 0; tti_d < tti_sizes.size(); tti_d++) {
				curr_sizes.push_back(tti_sizes[tti_d]);
			}


			FullTensor<zlx> t(curr_sizes.size(), curr_sizes.begin(), tmp_rx_tx2_tti);

			for (int i = 0; i < n_rx + n_tx_onedim_2 + tti_cut; i++) {
				tmp_rx_tx2_tti_factors[i] = d(gen); 
			}

			int tmp_iter_limit = 100;
			CanonicalTensor<zlx> t2(t, 1, tmp_rx_tx2_tti_factors, 1.0e-13, tmp_iter_limit, SALS);

			int fact_num = 0;
			int d_sum = 0;
			for (int d = 0; d < (int)rx_sizes.size(); d++) {
				for (int i = 0; i < (int)rx_sizes[d]; i++) {
					tmp_rx[i + r * rx_sizes[d] + rank * d_sum] = t2.get_factor(fact_num)[i];
				}
				d_sum += rx_sizes[d];
				fact_num++;
			}

			d_sum = 0;
			for (int d = 0; d < (int)tx_sizes_2.size(); d++) {
				for (int i = 0; i < (int)tx_sizes_2[d]; i++) {
					tmp_tx2[i + r * tx_sizes_2[d] + rank * d_sum] = t2.get_factor(fact_num)[i];
				}
				d_sum += tx_sizes_2[d];
				fact_num++;
			}

			d_sum = 0;
			for (int d = 0; d < (int)tti_sizes.size(); d++) {
				for (int i = 0; i < (int)tti_sizes[d]; i++) {
					tmp_tti[i + r * tti_sizes[d] + rank * d_sum] = t2.get_factor(fact_num)[i];
				}
				d_sum += tti_sizes[d];
				fact_num++;
			}
		}

		int iactr = 0, tmp_tti_ptr;
		double dimmlt;
		int ptr = 0;

		iactr_rx = 0;
		ptr = 0;
		for (int d = 0; d < (int)rx_sizes.size(); d++) {
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < (int)rx_sizes[d]; i++) {
					iappr[iactr] = tmp_rx[ptr];
					iactr++;
					ptr++;
				}
			}
		}
		iactr_tx_1 = iactr;

		// initial approximation loading -- tx
		dimmlt = 1.0;

		for (int d = 0; d < (int)tx_splash_1.size(); d++) {
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < (int)tx_splash_1[d]; i++) {
					iappr[iactr].real(cos(tx_phases_1[r] * i * dimmlt));
					iappr[iactr].imag(sin(tx_phases_1[r] * i * dimmlt));

					iactr++;
				};
			};

			dimmlt *= tx_splash_1[d];
		};

		if (!quantize_tx) {
			CanonicalTensor<zlx> t_tmp(tx_splash_1, rank, iappr + iactr_tx_1);
			iactr = iactr_tx_1;
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < n_tx_onedim_1; i++) {
					iappr[iactr] = t_tmp.get_element(i);
					iactr++;
				}
			}
		}
		iactr_tx_2 = iactr;

		ptr = 0;
		for (int d = 0; d < (int)tx_sizes_2.size(); d++) {
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < (int)tx_sizes_2[d]; i++) {
					iappr[iactr] = tmp_tx2[ptr];
					iactr++;
					ptr++;
				}
			}
		}
		iactr_sc = iactr;

		// initial approximation loading -- sc
		dimmlt = 1.0;

		for (int d = 0; d < (int)sc_splash.size(); d++) {
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < (int)sc_splash[d]; i++) {
					iappr[iactr].real(cos(sc_phases[r] * i * dimmlt));
					iappr[iactr].imag(sin(sc_phases[r] * i * dimmlt));

					iactr++;
				}
			}

			dimmlt *= sc_splash[d];
		}

		if (!quantize_sc) {
			CanonicalTensor<zlx> t_tmp(sc_splash, rank, iappr + iactr_sc);
			iactr = iactr_sc;
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < n_sc; i++) {
					iappr[iactr] = t_tmp.get_element(i);
					iactr++;
				}
			}

		}
		iactr_tti = iactr;

		ptr = 0;
		for (int d = 0; d < (int)tti_sizes.size(); d++) {
			for (int r = 0; r < rank; r++) {
				for (int i = 0; i < (int)tti_sizes[d]; i++) {
					iappr[iactr] = tmp_tti[ptr];
					iactr++;
					ptr++;
				}
			}
		}
// *****************************************************************************  cycle by TTI
		for (int step = 0; step < tti_steps; step++) {
			printf("TTI interval %d: [%d:%d]: \n", step + 1, tti_cut * step, tti_cut * (step + 1));

			for (int rx = 0; rx < n_rx; rx++) {
				for (int tti = 0; tti < tti_cut; tti++) {
					for (int sc = 0; sc < n_sc; sc++) {
						for (int tx_1 = 0; tx_1 < n_tx_onedim_1; tx_1++) {
							for (int tx_2 = 0; tx_2 < n_tx_onedim_2; tx_2++) {
								tensor_v[rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (sc + n_sc * tti)))] =
									all_t[rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (tx_two + n_tx_twodim * (sc + n_sc * (tti + step * tti_cut)))))];
							}
						}
					}
				}
			}

			FullTensor<zlx> t(parafac_sizes.size(), parafac_sizes.begin(), tensor_v.data());

			uint64_t parafac_rank = rank;
			int iter_limit = tailALS_iterations;

			if (step == 0) iter_limit = firstALS_iterations;

			sprintf(refname, "polar%d_step%d_initial", tx_two, step);
			QuantizedFactorFullScalePrint(refname, rx_sizes, tx_sizes_1, tx_sizes_2, sc_sizes, tti_sizes, rank, iappr);
/*
                // My addiion
            if( (ue==0) && (tx_two == 0) && (step == 0)){
    char out_fname3[240];
    sprintf(out_fname3, "iappr.txt");
    FILE* out_fd3 = fopen(out_fname3, "a");

    //save data3
    for(int kk=0;kk<216;kk++){
        fprintf(out_fd3, "%d\t%f\t%f\n", kk, iappr[kk]);
    }

    fclose(out_fd3);
    fflush(stdout);
            }
*/
                // End of my addition

			CanonicalTensor<zlx> t2(t, parafac_rank, iappr, 1.0e-13, iter_limit, SALS);
			CanonicalTensor<zlx> t3(parafac_sizes, parafac_rank, iappr);

			/*
			zlx * useme;
			useme = iappr;
			if (step == 0) 
			{
				iter_limit = firstALS_iterations;
				useme = NULL;
			}

			CanonicalTensor<zlx> t2(t, parafac_rank, useme, 1.0e-13, iter_limit, SALS);
			CanonicalTensor<zlx> t3(parafac_sizes, parafac_rank, iappr);
			*/

			for (int rx = 0; rx < n_rx; rx++) {
				for (int tti = 0; tti < tti_cut; tti++) {
					for (int sc = 0; sc < n_sc; sc++) {
						for (int tx_1 = 0; tx_1 < n_tx_onedim_1; tx_1++) {
							for (int tx_2 = 0; tx_2 < n_tx_onedim_2; tx_2++) {
								all_h[rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (tx_two + n_tx_twodim * (sc + n_sc * (tti + step * tti_cut)))))] =
									t2.get_element(rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (sc + n_sc * tti))));
							}
						}
					}
				}
			}

			// to calculate extrapolation efficiency:
			if (step > 0) {
				for (int rx = 0; rx < n_rx; rx++) {
					for (int tti = 0; tti < tti_cut; tti++) {
						for (int sc = 0; sc < n_sc; sc++) {
							for (int tx_1 = 0; tx_1 < n_tx_onedim_1; tx_1++) {
								for (int tx_2 = 0; tx_2 < n_tx_onedim_2; tx_2++) {
									all_e[rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (tx_two + n_tx_twodim * (sc + n_sc * (tti + step * tti_cut)))))] =
										t3.get_element(rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (sc + n_sc * tti))));
								}
							}
						}
					}
				}
			}

			// saving the obtained factors as the next initializer
			iactr = 0;

//            std::cout<<"tx_two "<<tx_two<<"  "<<n_tx_twodim<<std::endl;
            
//            std::cout<<"step "<<step<<"  "<<tti_steps<<std::endl;
            
			for (int d = 0; d < (int)parafac_sizes.size(); d++) {
                if(step==0)
                std::cout<<d<<"  "<<(int)parafac_sizes[d]<<std::endl;

				for (int r = 0; r < rank; r++) {
					for (int i = 0; i < (int)parafac_sizes[d]; i++) {
						iappr[iactr] = t2.get_factor(d)[i + r * parafac_sizes[d]]; // scalforsaving[0].real();

                        if(step==0){
                            if(!tx_two){
                                results_tensor[iactr] = iappr[iactr];

                            }else{
                                results_tensor[magic_size+iactr] = iappr[iactr]; // scalforsaving[0].real();
                            }
                        }

						iactr++;
					};
				};
			};
            std::cout<<step<<" step ;;;;  get_dimensionality "<<t2.get_dimensionality()<<"  parafac_sizes.size()= "<<(int)parafac_sizes.size()<<"  iactr= "<<iactr<<std::endl;

            /// Place to cout ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
                // My addiion
            if( (ue==0) && (tx_two == 0) && (step == 0)){
    char out_fname3[240];
    sprintf(out_fname3, "iappr.txt");
    FILE* out_fd3 = fopen(out_fname3, "a");

    //save data
    			int my_iactr = 0;
			for (int d = 0; d < (int)parafac_sizes.size(); d++) {
				for (int r = 0; r < rank; r++) {
					for (int i = 0; i < (int)parafac_sizes[d]; i++) {
                        fprintf(out_fd3, "%d\t%d\t%d\t%f\t%f\n", d, r, i,iappr[my_iactr]);
						my_iactr++;
					};
				};
			};


    fclose(out_fd3);
    fflush(stdout);
            }

                // My addiion 2
            if( (ue==0) && (tx_two == 0) && (step == 1)){
    char out_fname3[240];
    sprintf(out_fname3, "iappr2.txt");
    FILE* out_fd3 = fopen(out_fname3, "a");

    //save data
    			int my_iactr = 0;
			for (int d = 0; d < (int)parafac_sizes.size(); d++) {
				for (int r = 0; r < rank; r++) {
					for (int i = 0; i < (int)parafac_sizes[d]; i++) {
                        fprintf(out_fd3, "%d\t%d\t%d\t%f\t%f\n", d, r, i,iappr[my_iactr]);
						my_iactr++;
					};
				};
			};


    fclose(out_fd3);
    fflush(stdout);
            }

            
                // My addiion 3
            if( (ue==0) && (tx_two == 0) && (step == 2)){
    char out_fname3[240];
    sprintf(out_fname3, "iappr3.txt");
    FILE* out_fd3 = fopen(out_fname3, "a");

    //save data
    			int my_iactr = 0;
			for (int d = 0; d < (int)parafac_sizes.size(); d++) {
				for (int r = 0; r < rank; r++) {
					for (int i = 0; i < (int)parafac_sizes[d]; i++) {
                        fprintf(out_fd3, "%d\t%d\t%d\t%f\t%f\n", d, r, i,iappr[my_iactr]);
						my_iactr++;
					};
				};
			};


    fclose(out_fd3);
    fflush(stdout);
            }
*/
			sprintf(refname, "polar%d_step%d_afterALS", tx_two, step);
			QuantizedFactorFullScalePrint(refname, rx_sizes, tx_sizes_1, tx_sizes_2, sc_sizes, tti_sizes, rank, iappr);

			// ================================================================
			// tensor approximation:

			// modify rx factor ---------------------------------------------
			int iactr_start = iactr_rx;
			int* splash_rx;
			int splash_size;
			if (quantize_rx) {
				splash_size = rx_splash.size();
				splash_rx = new int[splash_size];
				for (int i = 0; i < splash_size; i++) {
					splash_rx[i] = rx_splash[i];
				}
			} else {
				splash_size = 1;
				splash_rx = new int[splash_size];
				splash_rx[0] = n_rx;
			}
			double * common_coef_1_rx = new double[rank];
			double * common_coef_2_rx = new double[rank];  
			int * num_saved_rx = new int[rank*2];  // for both amplitudes and phases
			if (rx_format==1) {				
				poly_steering(iappr, &iactr_start, splash_rx, &splash_size, &rank, 1, false, true, NUM_LEVELS_RX, common_coef_1_rx, common_coef_2_rx, false, num_saved_rx); // <-- iappr is changed!!
			}
			else {
				take_out_coef_noapp(iappr, &iactr_start, splash_rx, &splash_size, &rank, true, NUM_LEVELS_RX, common_coef_1_rx, common_coef_2_rx);  // <-- iappr is changed!!
			}
			delete[] num_saved_rx;
			delete[] splash_rx;          


			// modify tx factor ---------------------------------------------    
			iactr_start = iactr_tx_1;
			int* splash_tx_1;
			double * common_coef_1_tx_1 = new double[rank];
			double * common_coef_2_tx_1 = new double[rank];  
			int * num_saved_tx_1 = new int[rank*2];  // for both amplitudes and phases
			if (quantize_tx) {
				splash_size = tx_splash_1.size();
				splash_tx_1 = new int[splash_size];
				for (int i = 0; i < splash_size; i++) {
					splash_tx_1[i] = tx_splash_1[i];
				}
				if (tx_format==1) {				
					poly_steering(iappr, &iactr_start, splash_tx_1, &splash_size, &rank, tx_order, false, true, NUM_LEVELS_TX, common_coef_1_tx_1, common_coef_2_tx_1, save_flag, num_saved_tx_1); // <-- iappr is changed!!
					if (save_flag) {
						for (int k=0; k<rank; k++) {num_saved_tx1_total += num_saved_tx_1[k];}
						for (int k=rank; k<2*rank; k++) {num_saved_amplitudes_total += num_saved_tx_1[k];}
					}
				}
				else {
					take_out_coef_noapp(iappr, &iactr_start, splash_tx_1, &splash_size, &rank, true, NUM_LEVELS_TX, common_coef_1_tx_1, common_coef_2_tx_1);  // <-- iappr is changed!!
				}
			} else {
				splash_size = 1;
				splash_tx_1 = new int[splash_size];
				splash_tx_1[0] = n_tx_onedim_1;  
				if (tx_format==1) {				
					poly_steering(iappr, &iactr_start, splash_tx_1, &splash_size, &rank, tx_order, false, true, NUM_LEVELS_TX, common_coef_1_tx_1, common_coef_2_tx_1, save_flag, num_saved_tx_1); // <-- iappr is changed!!
					if (save_flag) {
						for (int k=0; k<rank; k++) {num_saved_tx1_total += num_saved_tx_1[k];}
						for (int k=rank; k<2*rank; k++) {num_saved_amplitudes_total += num_saved_tx_1[k];}
					}
				}
				else {
					take_out_coef_noapp(iappr, &iactr_start, splash_tx_1, &splash_size, &rank, true, NUM_LEVELS_TX, common_coef_1_tx_1, common_coef_2_tx_1);  // <-- iappr is changed!!
				}
			}
			delete[] num_saved_tx_1;
			delete[] splash_tx_1;     

			iactr_start = iactr_tx_2;
			int* splash_tx_2;
			double * common_coef_1_tx_2 = new double[rank];
			double * common_coef_2_tx_2 = new double[rank];
			int * num_saved_tx_2 = new int[rank*2];  // for both amplitudes and phases  
			if (quantize_tx) {
				splash_size = tx_splash_2.size();
				splash_tx_2 = new int[splash_size];
				for (int i = 0; i < splash_size; i++) {
					splash_tx_2[i] = tx_splash_2[i];
				}
				if (tx_format==1) {				
					poly_steering(iappr, &iactr_start, splash_tx_2, &splash_size, &rank, tx_order, false, true, NUM_LEVELS_TX, common_coef_1_tx_2, common_coef_2_tx_2, save_flag, num_saved_tx_2); // <-- iappr is changed!!
					if (save_flag) {
						for (int k=0; k<rank; k++) {num_saved_tx2_total += num_saved_tx_2[k];}
						for (int k=rank; k<2*rank; k++) {num_saved_amplitudes_total += num_saved_tx_2[k];}
					}
				}
				else {
					take_out_coef_noapp(iappr, &iactr_start, splash_tx_2, &splash_size, &rank, true, NUM_LEVELS_TX, common_coef_1_tx_2, common_coef_2_tx_2);  // <-- iappr is changed!!
				}
			} else {
				splash_size = 1;
				splash_tx_2 = new int[splash_size];
				splash_tx_2[0] = n_tx_onedim_2;  
				if (tx_format==1) {				
					poly_steering(iappr, &iactr_start, splash_tx_2, &splash_size, &rank, tx_order, false, true, NUM_LEVELS_TX, common_coef_1_tx_2, common_coef_2_tx_2, save_flag, num_saved_tx_2); // <-- iappr is changed!!
					if (save_flag) {
						for (int k=0; k<rank; k++) {num_saved_tx2_total += num_saved_tx_2[k];}
						for (int k=rank; k<2*rank; k++) {num_saved_amplitudes_total += num_saved_tx_2[k];}
					}
				}
				else {
					take_out_coef_noapp(iappr, &iactr_start, splash_tx_2, &splash_size, &rank, true, NUM_LEVELS_TX, common_coef_1_tx_2, common_coef_2_tx_2);  // <-- iappr is changed!!
				}
			}
			delete[] num_saved_tx_2;
			delete[] splash_tx_2;   


			// modify sc factor ------------------------------------------------
			iactr_start=iactr_sc;
			int* splash_sc;
			if (quantize_sc) {
				splash_size = sc_splash.size();
				splash_sc = new int[splash_size];
				for (int i = 0; i < splash_size; i++) {
					splash_sc[i] = sc_splash[i];
				}
			} else {
				splash_size = 1;
				splash_sc = new int[splash_size];
				splash_sc[0] = n_sc;
			}
			double * common_coef_1_sc = new double[rank];
			double * common_coef_2_sc = new double[rank];
			int * num_saved_sc = new int[rank*2];  // for both amplitudes and phases
			if (sc_format==1) 
			{			
				poly_steering(iappr, &iactr_start, splash_sc, &splash_size, &rank, 1, false, true, NUM_LEVELS_SC, common_coef_1_sc, common_coef_2_sc, save_flag, num_saved_sc); // <-- iappr is changed!!
				if (save_flag) {for (int k=0; k<rank; k++) {num_saved_sc_total += num_saved_sc[k];}}
				if (save_flag) {for (int k=rank; k<2*rank; k++) {num_saved_amplitudes_total += num_saved_sc[k];}}
			}
			else
			{
				take_out_coef_noapp(iappr, &iactr_start, splash_sc, &splash_size, &rank, true, NUM_LEVELS_SC, common_coef_1_sc, common_coef_2_sc);  // <-- iappr is changed!!
			}
			
			delete[] num_saved_sc;
			delete[] splash_sc;

			int num_saved_phases=0;
			int num_saved_amplitudes=0;


			// rank-one splashed ALS for each tti factor
			iactr = iactr_tti;
			if (!quantize_tti) {
				for (int r = 0; r < rank; r++) {
					printf("TTI-splashing tti factor %d.\n", r);
					for (int i = 0; i < tti_cut; i++) {
						ttifactor_v[i] = iappr[iactr];
						iactr++;
					};

					FullTensor<zlx> tfac(tti_splash.size(), tti_splash.begin(), ttifactor_v.data());
					CanonicalTensor<zlx> cfac(tfac, 1, NULL, 1.0e-13, tailALS_iterations, SALS);

					iactr -= tti_cut;
					for (int i = 0; i < tti_cut; i++) {
						iappr[iactr] = cfac.get_element(i);
						ttifactor_full[i+r*tti_cut] = iappr[iactr];
						iactr++;
					};
				};
			} else {
				for(int d = 0; d < tti_splash.size(); d++)
				{
					for(int r = 0; r < rank; r++)
					{
						for(int i = 0; i < tti_splash[d]; i++)
						{
							ttifactor_full[iactr-iactr_tti]=iappr[iactr];
							iactr++;
						};
					};
				};
			}

			double * common_coef_1_tti = new double[rank];
			double * common_coef_2_tti = new double[rank];
			int * num_saved_tti = new int[rank*2];
			int * splash_tti;
			if (quantize_tti) {
				splash_size = tti_splash.size();
				splash_tti = new int[splash_size];
				for (int i = 0; i < splash_size; i++) {
					splash_tti[i] = tti_splash[i];
				}
			} else {
				splash_size = 1;
				splash_tti = new int[splash_size];
				splash_tti[0] = tti_cut;
			}
			if (tti_order>0) 
			{
				poly_steering(iappr, &iactr_tti, splash_tti, &splash_size, &rank, tti_order, false, true, NUM_LEVELS_TTI, common_coef_1_tti, common_coef_2_tti, false, num_saved_tti); // <-- iappr is changed!!
			} 
			else
			{
				take_out_coef_noapp(iappr, &iactr_tti, splash_tti, &splash_size, &rank, true, NUM_LEVELS_TTI, common_coef_1_tti, common_coef_2_tti);
			}
			// multiply factors by common coefficient
			float * common_coef_1 = new float[rank];
			float * common_coef_2 = new float[rank];
			
			for(int r=0; r<rank; r++)
			{
				common_coef_1[r]=(float) flt_by_range((common_coef_1_tti[r]+common_coef_1_sc[r]+common_coef_1_tx_1[r]+common_coef_1_tx_2[r]+common_coef_1_rx[r]),-9.0,-3.0);
				common_coef_2[r]=(float) rounder((common_coef_2_tti[r]+common_coef_2_sc[r]+common_coef_2_tx_1[r]+common_coef_2_tx_2[r]+common_coef_2_rx[r]), NUM_LEVELS_COMMON); //rounder
			}
			multiply_factor_common_coef(iappr, &iactr_tti, splash_tti, &splash_size, &rank, common_coef_1, common_coef_2);

            
			// multiply factors:
			CanonicalTensor<zlx> t4(t, parafac_rank, iappr, 1.0e-13, 0, SALS);

			// approximated tensor:
			for (int rx = 0; rx < n_rx; rx++) {
				for (int tti = 0; tti < tti_cut; tti++) {
					for (int sc = 0; sc < n_sc; sc++) {
						for (int tx_1 = 0; tx_1 < n_tx_onedim_1; tx_1++) {
							for (int tx_2 = 0; tx_2 < n_tx_onedim_2; tx_2++) {
								all_i[rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (tx_two + n_tx_twodim * (sc + n_sc * (tti + step * tti_cut)))))] =
									t4.get_element(rx + n_rx * (tx_1 + n_tx_onedim_1 * (tx_2 + n_tx_onedim_2 * (sc + n_sc * tti))));
							}
						};
					};
				};
			}

			// extrapolation
			// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

			// return tti factor to extrapolate it:
			iactr = iactr_tti;
			if (!quantize_tti) {
				for(int r = 0; r < rank; r++)
				{
					for(int i = 0; i < tti_cut; i++)
					{
						iappr[iactr]=ttifactor_full[i+r*tti_cut];
						iactr++;
					};
				};	
			} else {
				for(int d = 0; d < tti_splash.size(); d++)
				{
					for(int r = 0; r < rank; r++)
					{
						for(int i = 0; i < tti_splash[d]; i++)
						{
							iappr[iactr]=ttifactor_full[iactr-iactr_tti];
							iactr++;
						};
					};
				};
			}


			// get, extrapolate and save tti factor
			if (tti_order>0) 
			{
				poly_steering(iappr, &iactr_tti, splash_tti, &splash_size, &rank, tti_order, true, true, NUM_LEVELS_TTI, common_coef_1_tti, common_coef_2_tti, false, num_saved_tti); // <-- iappr is changed!!
			}
			else
			{
				take_out_coef_noapp(iappr, &iactr_tti, splash_tti, &splash_size, &rank, true, NUM_LEVELS_TTI, common_coef_1_tti, common_coef_2_tti);
			}
			// multiply factors by common coefficient
			for(int r=0; r<rank; r++)
			{
				common_coef_1[r]=(float) flt_by_range((common_coef_1_tti[r]+common_coef_1_sc[r]+common_coef_1_tx_1[r]+common_coef_1_tx_2[r]+common_coef_1_rx[r]),-9.0,-3.0);
				common_coef_2[r]=(float) rounder((common_coef_2_tti[r]+common_coef_2_sc[r]+common_coef_2_tx_1[r]+common_coef_2_tx_2[r]+common_coef_2_rx[r]), NUM_LEVELS_COMMON); 
			}
			multiply_factor_common_coef(iappr, &iactr_tti, splash_tti, &splash_size, &rank, common_coef_1, common_coef_2);
			delete[] num_saved_tti;
			delete[] splash_tti;

			delete[] common_coef_1;
			delete[] common_coef_2;
			
			delete[] common_coef_1_rx;
			delete[] common_coef_2_rx;

			delete[] common_coef_1_tx_1;
			delete[] common_coef_2_tx_1;

			delete[] common_coef_1_tx_2;
			delete[] common_coef_2_tx_2;	

			delete[] common_coef_1_sc;
			delete[] common_coef_2_sc;
			
			delete[] common_coef_1_tti;
			delete[] common_coef_2_tti;

			// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		}

	}

	delete[] iappr;
	delete[] iappr_long;

	delete[] tx_phases_1;
	delete[] sc_phases;


	// =========================================================================
	// ============= Error metrics calculation =================================

	int n_tx_onedim = n_tx_onedim_1 * n_tx_onedim_2;
	int num_tx = num_tx_1 + num_tx_2;

	// Frobenius norm - Extrapolation, Approximation, ALS (Performance)
	int index;
	double up_sum_e=0;
	double down_sum=0;
	double up_sum_i=0;
	double up_sum_h=0;
	for(int tti = tti_cut*1; tti < n_tti; tti++)
	{
		for(int tx_two = 0; tx_two < n_tx_twodim; tx_two++)
		{
			for(int tx = 0; tx < n_tx_onedim; tx++)
			{
				for(int sc = 0; sc < n_sc; sc++)
				{
					for(int rx = 0; rx < n_rx; rx++)
					{
						index=rx + n_rx * (tx + n_tx_onedim * (tx_two + n_tx_twodim * (sc + n_sc * (tti ))));
						up_sum_e += std::norm(all_e[index]-all_t[index]);
						up_sum_i += std::norm(all_i[index]-all_t[index]);
						up_sum_h += std::norm(all_h[index]-all_t[index]);
						down_sum += std::norm(all_t[index]);
					}
				}
			};
		};
	};
	up_sum_e=std::sqrt(up_sum_e/down_sum);
	printf("Relative Frobenius norm, Extrapolation:  %.8f \n", up_sum_e);

	up_sum_i=std::sqrt(up_sum_i/down_sum);
	printf("Relative Frobenius norm, Approximation:  %.8f \n", up_sum_i);

	up_sum_h=std::sqrt(up_sum_h/down_sum);
	printf("Relative Frobenius norm, Performance:  %.8f \n", up_sum_h);

	// Average cosines
	double * iangle_avg = new double[n_rx];
	double * angle_avg = new double[n_rx];
	double * eangle_avg = new double[n_rx];
	double * cos_curr = new double[n_rx];

	for(int rx = 0; rx < n_rx; rx++)
	{
		iangle_avg[rx] = 0.0;
		angle_avg[rx] = 0.0;
		eangle_avg[rx] = 0.0;
	};

	for(int tti = tti_cut; tti < n_tti; tti++)
	{
		for(int sc = 0; sc < n_sc; sc++)
		{
			grass_cosine_row(n_rx, n_tx, all_t + n_tx * n_rx * (sc + n_sc * tti), all_e + n_tx * n_rx * (sc + n_sc * tti), cos_curr);
			for(int rx = 0; rx < n_rx; rx++)
			{
				double tmp = cos_curr[rx];
				if (tmp > 1.0) tmp = 1.0;

				eangle_avg[rx] += acos(tmp) / ((n_tti - tti_cut) * n_sc);

			};
		};
	};

	for(int tti = 0; tti < n_tti; tti++)
	{
		for(int sc = 0; sc < n_sc; sc++)
		{
			grass_cosine_row(n_rx, n_tx, all_t + n_tx * n_rx * (sc + n_sc * tti), all_h + n_tx * n_rx * (sc + n_sc * tti), cos_curr);

			for(int rx = 0; rx < n_rx; rx++)
			{
				double tmp = cos_curr[rx];
				if (tmp > 1.0) tmp = 1.0;

				angle_avg[rx] += acos(tmp) / (n_tti * n_sc);

			};
		};
	};

	for(int tti = 0; tti < n_tti; tti++)
	{
		for(int sc = 0; sc < n_sc; sc++)
		{
			grass_cosine_row(n_rx, n_tx, all_t + n_tx * n_rx * (sc + n_sc * tti), all_i + n_tx * n_rx * (sc + n_sc * tti), cos_curr);

			for(int rx = 0; rx < n_rx; rx++)
			{
				double tmp = cos_curr[rx];
				if (tmp > 1.0) tmp = 1.0;

				iangle_avg[rx] += acos(tmp) / (n_tti * n_sc);

			};
		};
	};

	printf("\n Extrapolation average cosines: \n");
	for(int rx = 0; rx < n_rx; rx++)
	{
		printf("%.8f ", cos(eangle_avg[rx]));
	};
	printf("\n");

	printf("\n ALS average cosines: \n");
	for(int rx = 0; rx < n_rx; rx++)
	{
		printf("%.8f ", cos(angle_avg[rx]));
	};
	printf("\n");

	printf("\n Approximation average cosines: \n");
	for(int rx = 0; rx < n_rx; rx++)
	{
		printf("%.8f ", cos(iangle_avg[rx]));
	};
	printf("\n");

	delete[] iangle_avg;
	delete[] angle_avg;
	delete[] eangle_avg;
	delete[] cos_curr;


	// Spectral effectiveness
	zlx * ashp = new zlx[n_rx * n_rx];

	double * sv = new double[n_rx];
	double * sv2 = new double[n_tx];
	zlx * tau = new zlx[n_rx];

	double ifunc = 0.0;
	double func = 0.0;
	double efunc = 0.0;
	double ideal = 0.0;

	//neither all_t nor all_h nor all_i survive here
	for(int tti = 0; tti < n_tti; tti++)
	{
		for(int sc = 0; sc < n_sc; sc++)
		{
			if (tti >= tti_cut)
			{		
				zgesvd(&cN, &cO, &n_rx, &n_tx, all_t + n_tx * n_rx * (sc + n_sc * tti), &n_rx, sv, NULL, &n_rx, NULL, &n_tx, zwork, &lwork, dwork, &info);

				for(int rx = 0; rx < n_rx; rx++)
				{
					ideal += log2(sv[rx] * sv[rx] / noise + 1.0) / (n_sc * (n_tti - tti_cut));
				};

				//ALS results
				zgelqf(&n_rx, &n_tx, all_h + n_tx * n_rx * (sc + n_sc * tti), &n_rx, tau, zwork, &lwork, &info);
				zunglq(&n_rx, &n_tx, &n_rx, all_h + n_tx * n_rx * (sc + n_sc * tti), &n_rx, tau, zwork, &lwork, &info);

				zgemm(&cN, &cC, &n_rx, &n_rx, &n_tx, &zone, all_t + n_tx * n_rx * (sc + n_sc * tti), &n_rx, all_h + n_tx * n_rx * (sc + n_sc * tti), &n_rx, &zzero, ashp, &n_rx);

				for(int rx = 0; rx < n_rx; rx++)
				{
					zlx zsv = sv[rx];
					zscal(&n_rx, &zsv, ashp + rx, &n_rx);
				};

				zgesvd(&cN, &cN, &n_rx, &n_rx, ashp, &n_rx, sv2, NULL, &n_rx, NULL, &n_rx, zwork, &lwork, dwork, &info);

				for(int rx = 0; rx < n_rx; rx++)
				{
					func += log2(sv2[rx] * sv2[rx] / noise + 1.0) / (n_sc * (n_tti - tti_cut));
				};

				//Approximation results
				zgelqf(&n_rx, &n_tx, all_i + n_tx * n_rx * (sc + n_sc * tti), &n_rx, tau, zwork, &lwork, &info);
				zunglq(&n_rx, &n_tx, &n_rx, all_i + n_tx * n_rx * (sc + n_sc * tti), &n_rx, tau, zwork, &lwork, &info);

				zgemm(&cN, &cC, &n_rx, &n_rx, &n_tx, &zone, all_t + n_tx * n_rx * (sc + n_sc * tti), &n_rx, all_i + n_tx * n_rx * (sc + n_sc * tti), &n_rx, &zzero, ashp, &n_rx);

				for(int rx = 0; rx < n_rx; rx++)
				{
					zlx zsv = sv[rx];
					zscal(&n_rx, &zsv, ashp + rx, &n_rx);
				};

				zgesvd(&cN, &cN, &n_rx, &n_rx, ashp, &n_rx, sv2, NULL, &n_rx, NULL, &n_rx, zwork, &lwork, dwork, &info);

				for(int rx = 0; rx < n_rx; rx++)
				{
					ifunc += log2(sv2[rx] * sv2[rx] / noise + 1.0) / (n_sc * (n_tti - tti_cut));
				};	

				//extrpolation results
				zgelqf(&n_rx, &n_tx, all_e + n_tx * n_rx * (sc + n_sc * tti), &n_rx, tau, zwork, &lwork, &info);
				zunglq(&n_rx, &n_tx, &n_rx, all_e + n_tx * n_rx * (sc + n_sc * tti), &n_rx, tau, zwork, &lwork, &info);

				zgemm(&cN, &cC, &n_rx, &n_rx, &n_tx, &zone, all_t + n_tx * n_rx * (sc + n_sc * tti), &n_rx, all_e + n_tx * n_rx * (sc + n_sc * tti), &n_rx, &zzero, ashp, &n_rx);

				for(int rx = 0; rx < n_rx; rx++)
				{
					zlx zsv = sv[rx];
					zscal(&n_rx, &zsv, ashp + rx, &n_rx);
				};

				zgesvd(&cN, &cN, &n_rx, &n_rx, ashp, &n_rx, sv2, NULL, &n_rx, NULL, &n_rx, zwork, &lwork, dwork, &info);

				for(int rx = 0; rx < n_rx; rx++)
				{
					efunc += log2(sv2[rx] * sv2[rx] / noise + 1.0) / (n_sc * (n_tti - tti_cut));
				};
			};
		};
	};


	printf("Total ALS spectral efficiency: %.2f.\n", func);
	printf("Total extrapolation spectral efficiency: %.2f.\n", efunc);
	printf("Total approximation spectral efficiency: %.2f.\n", ifunc);
	printf("Ideal spectral efficiency: %.2f.\n", ideal);
	printf("Performance: %.2f%% (ALS), %.2f%% (extrapolation), %.2f%% (approximation).\n", func * 100.0 / ideal, efunc * 100.0 / ideal, ifunc * 100.0 / ideal);
	int bits_add=account_bits_phase_save(num_tx_1, num_tx_2, num_sc, num_saved_sc_total, num_saved_tx1_total, num_saved_tx2_total, num_saved_amplitudes_total);
	double my_bits = bit_calculator(&n_tx_twodim, &rank, &tti_cut, &tx_order, &tti_order, rx_format, tx_format, sc_format, num_rx, num_tx, num_sc, num_tti, bits_add, tti_steps);
    
    printf("Number of bits per tti: %.2f.\n", my_bits);
    
    // My addiion
    /*
    char out_fname[240];
    sprintf(out_fname, "out_data.txt");
    FILE* out_fd = fopen(out_fname, "a");

    //save data2
    fprintf(out_fd, "ue=%d\t%f\t%f\t%f\t%f\n", ue, ideal, func, ifunc, efunc);

    fclose(out_fd);
    fflush(stdout);

    // End of my addition

    char out_fname2[240];
    sprintf(out_fname, "bits_data.txt");
    FILE* out_fd2 = fopen(out_fname, "a");

    //save data
    fprintf(out_fd2, "ue=%d\t%f\n", ue, my_bits);

    fclose(out_fd2);
    fflush(stdout);
*/
    // End of my addition

	delete[] lsmatrix;
	delete[] lscoeff;

	delete[] zwork;
	delete[] dwork;
	delete[] sv;
	delete[] sv2;
	delete[] tau;

	delete[] ashp;

	delete[] all_t;
	delete[] all_h;
	delete[] all_i;
	delete[] all_e;

	delete[] tmp_rx_tx2_tti;
	delete[] tmp_rx_tx2_tti_factors;

	delete[] tmp_rx;
	delete[] tmp_tx2;
	delete[] tmp_tti;

    results_tensor[magic_size*2] = scalforsaving[0];
    delete[] scalforsaving;

    return results_tensor;
};
