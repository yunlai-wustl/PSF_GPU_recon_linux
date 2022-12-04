#include "config.h"
#include <iostream>
#include <fstream>

using namespace std;

Config::Config(const std::string filename)
{

	_filename = filename;

	fstream fs;
	fs.open(filename.c_str(), fstream::in);
	if (fs.is_open())
	{
		ParseFile(fs);
		fs.close();
	}
	else
	{
		cout << "File " << filename << " could not be opened." << endl;
		throw 1;
	}
}

Config::~Config(){

}


// go through the config file and store settings in map
void
Config::ParseFile(std::fstream& fs)
{

	while (!fs.eof())
	{
		// get the next line
		string line;
		getline(fs, line);

		// skip over commented or blank lines
		if (line[0] == '#' || line == "")
			continue;

		// find equals sign
		size_t equal_sign_pos = line.find("=");
		if (equal_sign_pos == string::npos) // no match, so skip line
			continue;

		// extract string to left of equals sign
		string lhs = line.substr(0, equal_sign_pos);

		// extract string to right of equals sign, up to any comments on the right side of the value
		string rhs;
		size_t pound_sign_pos = line.find("#");
		if (pound_sign_pos == string::npos)
			rhs = line.substr(equal_sign_pos + 1);
		else
			rhs = line.substr(equal_sign_pos + 1, pound_sign_pos - (equal_sign_pos + 1));

		// remove whitespaces from extreme left and right sides of lhs and rhs
		RemoveWhitespace(lhs);
		RemoveWhitespace(rhs);

		// add to map
		_setting_map[lhs] = rhs;
	}
}

// only remove whitespace immediately to the right and left of equals sign, and after value
void
Config::RemoveWhitespace(string& text)
{
	string whitespace_chars = " \t\f\v\n\r";

	// remove whitespace on left side of text
	size_t first_non_whitespace_pos = text.find_first_not_of(whitespace_chars);
	text.erase(0, first_non_whitespace_pos);

	// remove whitespace on right side of text
	size_t last_non_whitespace_pos = text.find_last_not_of(whitespace_chars);
	text.erase(last_non_whitespace_pos + 1);
}

void
Config::RequiredKeyNotFound(const string& key)
{
	cout << "The parameter " << key << " was not found in " << _filename << ", but is required." << endl;
	throw 1;
}

template <>
int
Config::ConvertFromString<int>(const string& str_val)
{
	return atoi(str_val.c_str());
}

template <>
float
Config::ConvertFromString<float>(const string& str_val)
{
	return (float)atof(str_val.c_str());
}

template <>
string
Config::ConvertFromString<string>(const string& str_val)
{
	return str_val;
}
