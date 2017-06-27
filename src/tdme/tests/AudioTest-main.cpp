#include <tdme/tests/AudioTest.h>

extern void init_jvm();
extern java::lang::StringArray* make_args(int args, char** argv);

int main(int argc, char** argv)
{
    init_jvm();
    
    ::tdme::tests::AudioTest::main(make_args(argc, argv));
    
    return 0;
}
