//command parser from https://github.com/tanakh/cmdline

#include <stdio.h>
#include <stdlib.h>


#include "..\Solution_Items\GATE_data_structure.h"
#include "..\Solution_Items\PETData.h"
#include "..\Solution_Items\PETDataList.h"
#include "..\Solution_Items\PETDataListInveon.h"
#include "..\Solution_Items\PETDataListGateCoincidence.h"

#include "..\Solution_Items\cmdline.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	PETData inputdata;
	PETData outputdata;
	
	string inputfilename, outputfilename;
	string fromformat, toformat;

	cmdline::parser cmdparser;
	
	cmdparser.add<string>("input", 'i', "input file name", true, "");
	cmdparser.add<string>("output", 'o', "output file name", true, "");
	cmdparser.add<string>("fromformat", 'f', "from fromat", true, "", cmdline::oneof<string>("gatelistmodecoincidence", "gatelistmodesingle", "inveon", "cardinal64bit"));
	cmdparser.add<string>("toformat", 't', "to fromat", true, "", cmdline::oneof<string>("compactlistmodecoincidence", "compactlistmodesingles", "compactlistmodecoincidencewithpos"));

	cmdparser.add("help", 0, "print this message");
	cmdparser.set_program_name("FileFormatConversion");

	bool ok = cmdparser.parse(argc, argv);

	if (argc == 1 || cmdparser.exist("help")){
		cerr << cmdparser.usage();
		return 0;
	}

	if (!ok){
		cerr << cmdparser.error() << endl << cmdparser.usage();
		return 0;
	}

	cout << cmdparser.get<string>("input") << endl;
	cout << cmdparser.get<string>("output") << endl;
	cout << cmdparser.get<string>("fromformat") << endl;
	cout << cmdparser.get<string>("toformat") << endl;

	for (size_t i = 0; i<cmdparser.rest().size(); i++)
		cout << "- " << cmdparser.rest()[i] << endl;

	inputfilename = cmdparser.get<string>("input");
	outputfilename = cmdparser.get<string>("output");
	fromformat = cmdparser.get<string>("fromformat");
	toformat = cmdparser.get<string>("toformat");
	
	if (fromformat == "gatelistmodecoincidence"){
		 inputdata = new PETDataListGateCoincidence(inputfilename);
	}


	return 0;
}




