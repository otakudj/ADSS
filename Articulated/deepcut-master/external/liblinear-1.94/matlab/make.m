% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix

Type = ver;
% This part is for OCTAVE
if(strcmp(Type(1).Name, 'Octave') == 1)
    mex libsvmread.c
    mex libsvmwrite.c
    mex train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/*.c
    mex predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/*.c
% This part is for MATLAB
% Add -largeArrayDims on 64-bit machines of MATLAB
else
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
    %mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims train.c linear_model_matlab.c ../linear.cpp ../tron.cpp "../blas/*.c"
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
end

