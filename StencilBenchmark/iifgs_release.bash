#!/bin/bash

((max_pixel_count= 40 * 1000000))

width=1
height=1

# Удаляем лог перед запуском
rm -f ./resources/output/timelog.txt

while (( $width * $height * 100000 <= $max_pixel_count )); do
    ./build/samples/Release/sample_program.exe "$width" "$height"
    if [[ $width -gt $height ]]; then
        ((height++))
    else
        ((width++))
    fi
done