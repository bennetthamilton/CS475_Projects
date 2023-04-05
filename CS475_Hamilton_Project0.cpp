// Name: Bennett Hamilton
// Date: 4/5/23
// Description: practice project using parallel programming to perform operations on arrays

#include <omp.h>
#include <stdio.h>
#include <math.h>

#define NUMT1	        1	// number of threads to use -- do once for 1 and once for 4
#define NUMT2           4        
#define SIZE       	16000	// array size -- you get to decide
#define NUMTRIES        20	// how many times to run the timing to get reliable timing data

float A[SIZE];
float B[SIZE];
float C[SIZE];

// performance function
// ref: https://www.w3schools.com/cpp/cpp_function_param.asp
float performance(int threads) {

        omp_set_num_threads( threads );

        double maxMegaMults = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {
                double time0 = omp_get_wtime( );

                #pragma omp parallel for
                for( int i = 0; i < SIZE; i++ )
                {
                        C[i] = A[i] * B[i];
                }

                double time1 = omp_get_wtime( );
                double megaMults = (double)SIZE/(time1-time0)/1000000.;
                if( megaMults > maxMegaMults )
                        maxMegaMults = megaMults;
        }

        fprintf( stderr, "For %d threads, Peak Performance = %8.2lf MegaMults/Sec\n", threads, maxMegaMults );  

        return maxMegaMults;
}

int
main( )
{
#ifdef   _OPENMP
	fprintf( stderr, "OpenMP version %d is supported here\n", _OPENMP );
#else
	fprintf( stderr, "OpenMP is not supported here\n" );
	exit( 0 );
#endif

	// inialize the arrays:
	for( int i = 0; i < SIZE; i++ )
	{
		A[i] = 1.;
		B[i] = 2.;
	}

        float perf1 = performance(NUMT1);      // one thread
        float perf2 = performance(NUMT2);      // four threads

        float S = perf2 / perf1;               // speedup value
        float Fp = (4./3.)*( 1. - (1./S) );

        fprintf( stderr, "Speedup = %4.2lf\n", S);      // print values
        //fprintf( stderr, "Parallel Fraction = %4.2lf\n", Fp);
        
	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"

	// Speedup = (Peak performance for 4 threads) / (Peak performance for 1 thread)

        return 0;
}