#include <pthread.h> // remember to include the pthread header
#include <stdio.h>
#include <stdlib.h>

// Define the number of threads to create
#define NUM_THREADS 20

void *hello(void *id);

int main(int argc, char *argv[])
{

int myid, worker, val;

pthread_t threads[NUM_THREADS];

// Create the threads and have them start executing the function hello()
for(worker = 0; worker < NUM_THREADS; worker++)
{
val = pthread_create(&threads[worker], NULL, hello, (void *) worker);
if(val != 0)
{
printf("Error creating thread #%d, val = %d\n", worker, val);
exit(-1);
}
}

pthread_exit(NULL);
}

void *hello(void *id)
{
int myid;
myid = (int) id;
printf("Hello World from thread %d\n", myid);
pthread_exit(NULL); //terminate the thread
}