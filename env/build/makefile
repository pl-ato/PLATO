CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -fPIC `python3 -m pybind11 --includes` 

SRC = $(shell ls ../*.cpp)
OBJ = $(patsubst %.cpp, %.o, $(notdir $(SRC)))

TARGET = mover

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $(TARGET)`python3-config --extension-suffix` $(OBJ)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC)

.PHONY : clean
clean:
	rm -rf $(OBJ) $(TARGET)`python3-config --extension-suffix`