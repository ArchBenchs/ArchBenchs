#define _CRT_SECURE_NO_WARNINGS
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>


using namespace std;

struct node {
    int offset;
    int len;
    char next;

    node(int o, int l, char n) : offset(o), len(l), next(n) {}
};

char* read_bin(FILE* file) {
    fseek(file, 0, 2);
    int len = ftell(file);
    fseek(file, 0, 0);
    char* t = new char[len+1];
    fread(t, 1, len, file);
    t[len] = '\0';
    return t;
}

void write_tokens(FILE* file, vector<node> tokens) {
    char* out = new char[tokens.size() * 4];
    int i = 0;
    for (const auto& t:tokens) {
        out[i++] = t.offset >> 8;
        out[i++] = t.offset & 0xff;
        out[i++] = t.len;
        out[i++] = t.next;
    }
    fwrite(out, 1, i, file);
}

vector<node> read_tokens(FILE* file) {
    vector<node> tokens;
    unsigned char t[4];
    while(fread(t, 1, 4, file))tokens.emplace_back((t[0] << 8) + t[1], t[2], t[3]);
    return tokens;
}

vector<node> compress(string in, int windowSize = 4096, int bufferSize = 15) {
    vector<node> tokens;
    time_t s, t=0;
    int length = size(in);
    vector<int> qwe(20,0);
    int pos = 0;

    while (pos < length) {
        for (int i = 0; i < 20; i++)qwe[i] = 0;
        s = clock();
        #pragma omp parallel for
        for (int i = max(0, pos - windowSize); i < pos; i++) {
            if (in[i] != in[pos])continue;
            int curLen = 0;
            while (curLen < bufferSize && pos + curLen < length && in[i + curLen] == in[pos + curLen]) curLen++;
            qwe[curLen] = pos-i;
        }
        t += clock() - s;
        int i = 17;
        qwe[0] = 0;
        while (qwe[i] == 0 && i>0)i--;
        char nextChar = (pos + i < length) ? in[pos + i] : '\0';
        tokens.emplace_back(qwe[i], i, nextChar);

        pos += i + 1;
    }
    cout << "length of match seek time " << t <<endl;
    return tokens;
}

char* decompress(const vector<node>& tokens) {
    int length = 0;
    for (int i = 0; i < tokens.size(); i++)length += tokens[i].len + 1;
    char* out=new char[length];
    int j = 0;
    for (const auto& token : tokens) {
        if (token.len > 0) { 
            int start = j - token.offset;
            for (int i = 0; i < token.len; ++i) {
                out[j] = out[start + i];
                j++;
            }
        }
        out[j]= token.next;
        j++;
    }

    return out;
}

int main() {
    FILE* in = fopen("C:\\Users\\dzhak\\Documents\\aboba\\1.txt", "rb");
    FILE* compressed = fopen("C:\\Users\\dzhak\\Documents\\aboba\\2.txt", "wb");
    FILE* decompressed = fopen("C:\\Users\\dzhak\\Documents\\aboba\\3.txt", "wb");
    FILE* io_check = fopen("C:\\Users\\dzhak\\Documents\\aboba\\4.txt", "wb");
    int mist1 = 0;
    int mist2 = 0;

    omp_set_num_threads(8);

    time_t s_comp, f_comp;//начало и конец компрессии
    time_t s_full, f_full;//начало и конец всей программы
    s_full = clock();

    string orig=read_bin(in);
    //for (int i = 0; i < length; i++)cout << orig[i] << " ";
    int length = size(orig);
    cout << length << endl;

    s_comp = clock();
    auto comp = compress(orig);
    f_comp = clock();
    cout <<"Compression time: "<< f_comp - s_comp << endl;
    //cout << "Compressed tokens:" << endl;
    //for (const auto& token : comp)cout <<  token.offset << "," << token.len << "," << (token.next ? token.next : '#') << " ";
    // //cout << endl;
    write_tokens(compressed,comp);
    fclose(compressed);
    char* decomp = decompress(comp);
    cout << "Compression rate: " <<(double)orig.size()/comp.size()/3<< endl;
    fwrite(decomp, 1, length, decompressed);

    FILE* qwe = fopen("C:\\Users\\dzhak\\Documents\\aboba\\2.txt", "rb");
    vector<node> t=read_tokens(qwe);
    char* decomp2 = decompress(t);
    fwrite(decomp2, 1, length, io_check);

    for (int i = 0; i < length; i++) {
        if(orig[i] != decomp[i])mist1++;
        if (orig[i] != decomp2[i])mist2++;
    }
    cout << "Number of mistakes:\ndecompressed: " << mist1 << "\nio_check: " << mist2 << endl;
    f_full = clock();
    cout <<"Full time: "<< f_full - s_full << endl;
}