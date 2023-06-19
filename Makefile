rwkv.cpp/bin/test_tiny_rwkv:
	cd rwkv.cpp && cmake .  && cmake --build .
	cd rwkv.cpp && cmake . -DRWKV_BUILD_SHARED_LIBRARY=OFF && cmake --build .

librwkv.a: rwkv.cpp/bin/test_tiny_rwkv
	cp rwkv.cpp/librwkv.a .
	cp rwkv.cpp/librwkv.so .

librwkv.so: rwkv.cpp/bin/test_tiny_rwkv

build: librwkv.a librwkv.so

clean:
	rm -rfv *.a
	rm -rfv examples/ai
	$(MAKE) -C rwkv.cpp clean

examples/ai: librwkv.a
	C_INCLUDE_PATH=$(shell pwd) LIBRARY_PATH=$(shell pwd) go build -o examples/ai ./examples
