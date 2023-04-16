#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// print debugging messages?
#ifndef DEBUG
#define DEBUG	false
#endif

// setting the number of threads:
// #ifndef NUMT
// #define NUMT		    2
// #endif

// setting the number of trials in the monte carlo simulation:
// #ifndef NUMTRIALS
// #define NUMTRIALS	50000
// #endif

// how many tries to discover the maximum performance:
#define NUMTIMES	20

#define CSV

// the pins; numbers are constants:
const float PinAx =	3.0f;
const float PinAy =	4.0f;
const float PinAr =	2.0f;

const float PinBx =	4.0f;
const float PinBy =	5.0f;
const float PinBr =	2.0f;

const float PinCx =	5.0f;
const float PinCy =	4.0f;
const float PinCr =	2.0f;


// ranges for the random numbers:

const float HoleAx   =  2.90f;
const float HoleAy   =  4.10f;
const float HoleAr   =  2.20f;
const float HoleAxPM =	0.20f;
const float HoleAyPM =	0.20f;
const float HoleArPM =	0.20f;

const float HoleBx   =  4.10f;
const float HoleBy   =  4.90f;
const float HoleBr   =  2.20f;
const float HoleBxPM =	0.10f;
const float HoleByPM =	0.10f;
const float HoleBrPM =	0.20f;

const float HoleCx   =  5.00f;
const float HoleCy   =  4.00f;
const float HoleCr   =  2.20f;
const float HoleCxPM =	0.10f;
const float HoleCyPM =	0.05f;
const float HoleCrPM =	0.20f;


// return a random number within a certain range:
float Ranf( float low, float high ) 
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

// call this at the top of your main( ) if you want to force your program to use
// a different random number sequence every time you run it:
void TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}


// square a number:
float Sqr( float x )
{
	return x*x;
}


// square root of the sum of the squares:
float Length( float dx, float dy )
{
	return  sqrt( Sqr(dx) + Sqr(dy) );
}

void run( int numt, int numtrials)
{
    #ifdef _OPENMP
        //fprintf( stderr, "OpenMP is supported -- version = %d\n", _OPENMP );
    #else
            fprintf( stderr, "No OpenMP support!\n" );
            return;
    #endif

            TimeOfDaySeed( );               // seed the random number generator

            omp_set_num_threads( numt );    // set the number of threads to use in parallelizing the for-loop:`

            // better to define these here so that the rand() calls don't get into the thread timing:
            float *holeaxs  = new float [numtrials];
            float *holeays  = new float [numtrials];
            float *holears  = new float [numtrials];

            float *holebxs  = new float [numtrials];
            float *holebys  = new float [numtrials];
            float *holebrs  = new float [numtrials];

            float *holecxs  = new float [numtrials];
            float *holecys  = new float [numtrials];
            float *holecrs  = new float [numtrials];

            // fill the random-value arrays:
            for( int n = 0; n < numtrials; n++ )
            {
                    holeaxs[n]  = Ranf(  HoleAx-HoleAxPM,  HoleAx+HoleAxPM );
                    holeays[n]  = Ranf(  HoleAy-HoleAyPM,  HoleAy+HoleAyPM );
                    holears[n]  = Ranf(  HoleAr-HoleArPM,  HoleAr+HoleArPM );

                    holebxs[n]  = Ranf(  HoleBx-HoleBxPM,  HoleBx+HoleBxPM );
                    holebys[n]  = Ranf(  HoleBy-HoleByPM,  HoleBy+HoleByPM );
                    holebrs[n]  = Ranf(  HoleBr-HoleBrPM,  HoleBr+HoleBrPM );


                    holecxs[n]  = Ranf(  HoleCx-HoleCxPM,  HoleCx+HoleCxPM );
                    holecys[n]  = Ranf(  HoleCy-HoleCyPM,  HoleCy+HoleCyPM );
                    holecrs[n]  = Ranf(  HoleCr-HoleCrPM,  HoleCr+HoleCrPM );
            }

            // get ready to record the maximum performance and the probability:
            double  maxPerformance = 0.;    // must be declared outside the NUMTIMES loop
            int     numSuccesses;           // must be declared outside the NUMTIMES loop


            // looking for the maximum performance:
            for( int times = 0; times < NUMTIMES; times++ )
            {
                    double time0 = omp_get_wtime( );

                    numSuccesses = 0;

            // note: the Pin numbers don't need to be declared shared( ) because they are const variables!
                    #pragma omp parallel for default(none) shared(holeaxs,holeays,holears, holebxs,holebys,holebrs, holecxs,holecys,holecrs, stderr, PinAx, PinAy, PinAr, PinBx, PinBy, PinBr, PinCx, PinCy, PinCr, numtrials) reduction(+:numSuccesses)
                    for( int n = 0; n < numtrials; n++ )
                    {
                        // randomize everything:
                        float holeax = holeaxs[n];
                        float holeay = holeays[n];
                        float holear = holears[n];

                        float holebx = holebxs[n];
                        float holeby = holebys[n];
                        float holebr = holebrs[n];

                        float holecx = holecxs[n];
                        float holecy = holecys[n];
                        float holecr = holecrs[n];

                        float da = Length( PinAx - holeax, PinAy - holeay );

                        if( da + PinAr <= holear )
                        {
                            float db = Length( PinBx - holebx, PinBy - holeby );
                            if( db + PinBr <= holebr )
                            {
                                float dc = Length( PinCx - holecx, PinCy - holecy );
                                if( dc + PinCr <= holecr ) numSuccesses++;
                            }
                        }

                    } // for( # of  monte carlo trials )

                    double time1 = omp_get_wtime( );
                    double megaTrialsPerSecond = (double)numtrials / ( time1 - time0 ) / 1000000.;
                    if( megaTrialsPerSecond > maxPerformance )
                            maxPerformance = megaTrialsPerSecond;

            } // for ( # of timing tries )

            float probability = (float)numSuccesses/(float)( numtrials );        // just get for last NUMTIMES run

    #ifdef CSV
            fprintf(stderr, "%2d, %8d, %6.2f, %6.2lf\n", numt, numtrials, 100.*probability, maxPerformance);
    #else
            fprintf(stderr, "%2d threads ; %8d trials ; probability = %6.2f ; megatrials/sec = %6.2lf\n",
                    numt, numtrials, 100.*probability, maxPerformance);
    #endif
}

int main( int argc, char *argv[ ] )
{

    int numThreads[] = {1, 2, 4, 8, 12, 16, 20, 24, 32};
    int numTrials[] = {1, 10, 100, 1000, 10000, 100000, 500000, 1000000};

    for( int thread : numThreads )
    {
        for( int trials : numTrials)
        {
            run(thread, trials);
        }
    } 

}