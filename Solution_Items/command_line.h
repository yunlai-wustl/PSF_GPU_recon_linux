#pragma once

#include <string>
using namespace std;

// simple commmand-line parser

namespace CommandLine
{
  string GetProgramName(char argv0[]);
  string GetValue(int argc, char* argv[], const string& option);
}
