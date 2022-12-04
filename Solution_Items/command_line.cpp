#include "command_line.h"

// remove path preceding executable name
string
CommandLine::GetProgramName(char argv0[])
{
  string program_name = argv0;
  size_t pos = program_name.find_last_of("/\\");
  program_name.erase(0, pos + 1);

  return program_name;
}

// get the string value associated with a command line switch/option
string
CommandLine::GetValue(int argc, char* argv[], const string& option)
{
  // loop with stride of 2 since option and value come in pairs
  for (int i = 1; i < argc; i += 2)
  {
    if ((string)argv[i] == option)
      return (string)argv[i+1];
  }

  return "";
}
