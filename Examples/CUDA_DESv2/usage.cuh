#include <iostream>
#include <algorithm>

void usage();
void get_parameters(int argc, char** argv, uint64_t *cipher, char **key_alphabet, int *key_length,
	char **message_alphabet, int *message_length, bool *cpu, bool *gpu);
char* getCmdOption(char ** begin, char ** end, const std::string & option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

void usage()
{
	std::cout << "CUDA DES Cracker\n";
	printf("\t--cipher <hex>                        Ciphertext to crack\n");
	printf("\t--key-alphabet <alphabet>             Alphabet of possible chars in the key\n");
	printf("\t--key-length <length (1-8 chars)>     Length of the key\n");
	printf("\t--message-alphabet <alphabet>         Alphabet of possible chars in the message\n");
	printf("\t--message-length <length (1-8 chars)> Length of the message\n");
	printf("\t--run-cpu [optional]                  Run CPU des cracker\n");
	printf("\t--run-gpu [optional]                  Run GPU des cracker\n");
	printf("\Example: ./des --cipher c5697270d51e09a4 --key-alphabet qadwczd --key-length 5 --message-alphabet kpmqaz --message-length 4 --run-cpu --run-gpu\n");
}

void get_parameters(int argc, char** argv, uint64_t *cipher, char **key_alphabet, int *key_length,
	char **message_alphabet, int *message_length, bool *cpu, bool *gpu)
{
	if(argc < 10)
	{
		usage();
		exit(1);
	}
	if(cmdOptionExists(argv, argv + argc, "--help"))
    {
		usage();
		exit(0);
    }
	if(cmdOptionExists(argv, argv + argc, "--run-cpu"))
    {
		*cpu = true;
    }
	if(cmdOptionExists(argv, argv + argc, "--run-gpu"))
    {
		*gpu = true;
    }

	*key_alphabet = getCmdOption(argv, argv + argc, "--key-alphabet");
	*key_length =  std::stoi(getCmdOption(argv, argv + argc, "--key-length"));
	*message_alphabet = getCmdOption(argv, argv + argc, "--message-alphabet");
	*message_length =  std::stoi(getCmdOption(argv, argv + argc, "--message-length"));
	char *cipher_hex = getCmdOption(argv, argv + argc, "--cipher");
	*cipher = strtoull(cipher_hex, nullptr, 16);
}