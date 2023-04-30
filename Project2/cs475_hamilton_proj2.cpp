// CS475 Project 2
// Bennett Hamilton
// 4/29/23

#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int	NowYear;	    	// 2023 - 2028
int	NowMonth;	    	// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// rye grass height in inches
int	NowNumRabbits;		// number of rabbits in the current population

unsigned int seed = 0;  // seed for random number generation

const float RYEGRASS_GROWS_PER_MONTH =		25.0;   // units are in inches
const float ONE_RABBITS_EATS_PER_MONTH =	 1.0;   

const float AVG_PRECIP_PER_MONTH =	       12.0;	// average (units are in inches)
const float AMP_PRECIP_PER_MONTH =	    	4.0;	// plus or minus
const float RANDOM_PRECIP =		        	2.0;	// plus or minus noise

const float AVG_TEMP =			        	60.0;	// average
const float AMP_TEMP =			        	20.0;	// plus or minus
const float RANDOM_TEMP =	        		10.0;	// plus or minus noise

const float MIDTEMP =			        	60.0;   // units are in fahrenheit
const float MIDPRECIP =		        		14.0;

float Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

float Sqr( float x )
{
    return x*x;
}

float TempFactor(float NowTemp)
{
    return exp(   -Sqr(  ( NowTemp - MIDTEMP ) / 10.  )   );
} 

float PrecipFactor(float NowPrecip)
{
    return exp(   -Sqr(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
}

void CalcEnvironment()
{
    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );

    if( NowPrecip < 0. ) NowPrecip = 0.;
}

void Rabbits()
{
    while( NowYear < 2029 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation: 

        int nextNumRabbits = NowNumRabbits;
        int carryingCapacity = (int)( NowHeight );

        if( nextNumRabbits < carryingCapacity )
            nextNumRabbits++;
        else if ( nextNumRabbits > carryingCapacity )
            nextNumRabbits--;

        if( nextNumRabbits < 0 ) nextNumRabbits = 0;

        // DoneComputing barrier:
        #pragma omp barrier

        NowNumRabbits = nextNumRabbits;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DoneFarming barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void RyeGrass()
{
    while( NowYear < 2029 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation: 

        float tempFactor = TempFactor(NowTemp);
        float precipFactor = PrecipFactor(NowPrecip);

         float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * RYEGRASS_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumRabbits * ONE_RABBITS_EATS_PER_MONTH;

        if( nextHeight < 0 ) nextHeight = 0;

        // DoneComputing barrier:
        #pragma omp barrier

        NowHeight = nextHeight;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DoneFarming barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void Watcher()
{
    while( NowYear < 2029 )
    {

        // DoneComputing barrier:
        #pragma omp barrier

        // DoneAssigning barrier:
        #pragma omp barrier

        // DoneFarming barrier:
        #pragma omp barrier

        // print results
        fprintf(stderr, "%3d, %4.2f, %3.1f, %3.2lf, %2d, %4d\n", NowNumRabbits, NowHeight, NowTemp, NowPrecip, NowMonth + 1, NowYear);
        
        if (NowMonth == 11){    // increment month and possibly year
            NowMonth = 0;
            NowYear ++;
        } 
        else {
            NowMonth ++;
        }

        CalcEnvironment();      // calculate environmental parameters

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void MyFarmer()
{
      while( NowYear < 2029 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation: 

        // DoneComputing barrier:
        

        // DoneAssigning barrier:
        #pragma omp barrier

        int nextR = NowNumRabbits;
        float nextH = NowHeight;
        float nextP = NowPrecip;
        
        if (NowHeight < 10) nextP += 10;                         // the farmer waters the rye grass field when its low
        if (NowHeight >= 75) nextH = 10;                         // and cuts the grass when it gets too tall
        if (NowNumRabbits <= 2 and NowHeight > 50) nextR += 4;   // and breeds more rabbits when there is only two left
        if (NowNumRabbits >= 15) nextR = nextR / 2;              // and sells rabbits when there are WAY too many
        
        if( nextR < 0 ) nextR = 0;
        if( nextH < 0 ) nextH = 0;
        NowNumRabbits = nextR;
        NowHeight = nextH;
        NowPrecip = nextP;
        
        // DoneFarming barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void RunSimulation()
{

    CalcEnvironment();      // calculate starting environmental factors

    #ifdef _OPENMP
        //fprintf( stderr, "OpenMP is supported -- version = %d\n", _OPENMP );
    #else
            fprintf( stderr, "No OpenMP support!\n" );
            return;
    #endif

        omp_set_num_threads( 4 );	// same as # of sections 
        #pragma omp parallel sections 
        {
            #pragma omp section
            {
                Rabbits( );
            }

            #pragma omp section
            {
                RyeGrass( );
            }

            #pragma omp section
            {
                MyFarmer( );
            }

            #pragma omp section
            {
                Watcher( );	
            }
        }       // implied barrier -- all functions must return in order
                // to allow any of them to get past here
}

int main()
{
    
    // starting date and time:
    NowMonth =    0;
    NowYear  = 2023;

    // starting state (feel free to change this if you want):
    NowNumRabbits = 1;
    NowHeight =  5.;

    RunSimulation();

}
