/**
 * 
 * The student is required to add content to this file.  This file is
 * your implementation of the project and will be submitted for grading.
 * 
 */

#include "main.h"
#include "student.h"

/**********************************************************************************
 * 
 * Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).
 * 
 **********************************************************************************/

/**********************************************************************************
 * 
 * Implement your utility functions here
 * 
 **********************************************************************************/




/**********************************************************************************
 * 
 * Implement the three main program functions
 * 
 **********************************************************************************/



/**
 * This function transfers data from Host to Device
 */
void host_to_dev()
{

}

/**
 * This function performs the bitonic sort and merge by calling the
 * kernels you have defined in the section above
 */
void bitonic_sort()
{

}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host()
{
    // Default value.  You can return any pointer you wish based on
    // your implementation.
    return arrSortedGpu;
}

/**
 * This function frees memory and anything else the student requires 
 * before exiting the program
 */
void cleanup(){
    
    // You may modify/remove these as needed to make your implementation work
    // properly. The defaults provided here allow the skeleton code to compile.    
    free(arrCpu);
    free(arrSortedGpu);
}
