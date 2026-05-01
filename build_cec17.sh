set -e   # stop on any error

echo "  Building CEC17 shared library for Python"
echo "************************************************"

echo "[1/3] Patching cec17_test_func.cpp for Linux..."

sed 's/#include <WINDOWS.H>/\/\/ Windows.h removed for Linux\/Mac/g; s/%Lf/%lf/g' \
    cec17_test_func.cpp > cec17_test_func_linux.cpp

echo "      Done — saved as cec17_test_func_linux.cpp"

echo "[2/3] Writing cec17_wrapper.c..."

cat > cec17_wrapper.c << 'CWRAPPER'
CWRAPPER

echo "      Done — cec17_wrapper.c written"

echo "[3/3] Compiling cec17.so..."

g++ -O2 -shared -fPIC \
    -o cec17.so \
    cec17_wrapper.c \
    cec17_test_func_linux.cpp \
    -lm

echo "      Done — cec17.so compiled"

echo ""
echo "  Build complete!  cec17.so is ready."
echo ""
echo "  Test the bridge by running:"
echo "      python cec17_bridge.py"
echo ""
echo "  Make sure your input_data/ folder is here:"
ls -d input_data 2>/dev/null && echo "      [OK] input_data/ found" || \
echo "      [!!] input_data/ NOT found — copy it here before running!"
echo "*****************************************************"