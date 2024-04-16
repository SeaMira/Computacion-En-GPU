#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <memory>


std::string load_from_file(const std::string &path) {

    auto close_file = [](FILE *f) { fclose(f); };

    auto holder = std::unique_ptr<FILE, decltype(close_file)>(fopen(path.c_str(), "rb"), close_file);
    if (!holder)
        return "";

    FILE *f = holder.get();

    // in C++17 following lines can be folded into std::filesystem::file_size invocation
    if (fseek(f, 0, SEEK_END) < 0)
        return "";

    const long size = ftell(f);
    if (size < 0)
        return "";

    if (fseek(f, 0, SEEK_SET) < 0)
        return "";

    std::string res;
    res.resize(size);

    // C++17 defines .data() which returns a non-const pointer
    fread(const_cast<char *>(res.data()), 1, size, f);

    return res;
}