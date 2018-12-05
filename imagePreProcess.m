function [J] = imagePreProcess(name_Img)
  I = imread(name_Img);
  I = double(I);
  I = rgb2gray(I);
  J = imresize(I, [20 20]);
  J = double(J);
endfunction
