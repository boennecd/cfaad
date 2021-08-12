#include <AAD.h>
#include <AADInit.hpp>

#define CATCH_CONFIG_MAIN 
#include <catch2/catch_all.hpp>

int main( int argc, char* argv[] )
{
    Catch::Session session;
    
    int returnCode = session.applyCommandLine( argc, argv );
    if(returnCode != 0) // Indicates a command line error
        return returnCode;
        
    return session.run();
}
