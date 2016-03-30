
CXX      = nvcc
LINK     = nvcc
#CXXFLAGS = -O3 -Xptxas -v -Xptxas -dlcm=ca -arch=sm_20 -m64 
#CXXFLAGS = -O3 -Xptxas -v -Xptxas -dlcm=ca -arch=sm_20 -m64 -DDOUBLE
CXXFLAGS = -O3 -Xptxas -v -Xptxas -dlcm=ca -arch=sm_20 -m64 -DPROFILE
#CXXFLAGS = -O3 -Xptxas -v -Xptxas -dlcm=ca -arch=sm_20 -m64 -DPROFILE -DDOUBLE
#CXXFLAGS = -g -DDEBUG_ATPG -Wall #Turn on debugging
INCPATH  =
LIBS     = -lreadline -lcurses
DEL_FILE = rm -f

####### Files

HEADERS = circuit.h gate.h gauss.h parser.h device.h 

SOURCES = circuit.cu main.cu gauss.cu parser.cu	device.cu

OBJECTS = circuit.o main.o gauss.o parser.o device.o

TARGET  = LDPC

####### Implicit rules

.SUFFIXES: .o .cpp .cc .cxx .cu

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

.cu.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

####### Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LIBS) $(INCPATH)

$(OBJECTS): $(HEADERS) Makefile

clean:
	@$(DEL_FILE) $(OBJECTS) $(TARGET)

