#ifndef CONFIG_H
#define CONFIG_H

#include <map>
#include "global.h"


using namespace std;

typedef std::map<std::string, std::string>::iterator map_iterator;

class Config{
public:
	Config(const std::string config_file_name);
	~Config();

	template <typename T>
	void GetValue(const string& key, T& value, bool required = true);

private:
		// map structure of the settings from the file
		// key and value are both of type string in this map
	std::map<std::string, std::string> _setting_map;

	

	std::string _filename;

	void ParseFile(std::fstream& fs);
	void RemoveWhitespace(std::string& text);

	template <typename T>
	T ConvertFromString(const std::string& str_val);

	void RequiredKeyNotFound(const std::string& key);

};


template <typename T>
void
Config::GetValue(const std::string &key, T& value, bool required)
{

	map_iterator iter = _setting_map.find(key);

	if (iter != _setting_map.end()) // if key was found
	{
		value = ConvertFromString<T>(iter->second);
	}
	else if (required)
	{
		RequiredKeyNotFound(key);
	}
}

#endif