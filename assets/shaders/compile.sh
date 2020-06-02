#!/bin/bash

for i in *.frag; do
  glslangValidator -V $i -o $i.spv
done

for i in *.vert; do
  glslangValidator -S vert -V $i -o $i.spv
done