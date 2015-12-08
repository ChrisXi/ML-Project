package PrincetonMatrix;

import java.util.Random;


/******************************************************************************
 *  Compilation:  javac Matrix.java
 *  Execution:    java Matrix
 *
 *  A bare-bones collection of static methods for manipulating
 *  matrices.
 *
 ******************************************************************************/

public class Matrix {

    // return a random m-by-n matrix with values between 0 and 1
    public static double[][] random(int m, int n) {
        double[][] C = new double[m][n];
        Random rand = new Random();	 
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = rand.nextGaussian();
        return C;
    }

    // return n-by-n identity matrix I
    public static double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++)
            I[i][i] = 1;
        return I;
    }

    // return x^T y
    public static double dot(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // return C = A^T
    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[j][i] = A[i][j];
        return C;
    }

    // return C = A + B
    public static double[][] add(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    // return C = A - B
    public static double[][] subtract(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }
    
    // return C = A - B
    public static double[] subtract(double[] A, double[] B) {
        int m = A.length;
        double[] C = new double[m];
        for (int i = 0; i < m; i++)
        	C[i] = A[i] - B[i];
        return C;
    }

    // return C = A * B
    public static double[][] multiply(double[][] A, double[][] B) {
        int mA = A.length;
        int nA = A[0].length;
        int mB = B.length;
        int nB = B[0].length;
        if (nA != mB) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] C = new double[mA][nB];
        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nB; j++)
                for (int k = 0; k < nA; k++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    // matrix-vector multiplication (y = A * x)
    public static double[] multiply(double[][] A, double[] x) {
        int m = A.length;
        int n = A[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += A[i][j] * x[j];
        return y;
    }

    // return 
    public static double[] multiply(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++)
            result[i] = x[i] * y[i];
        return result;
    }
    
    public static double[][] multiplyTwo(double[] x, double[] y){
    	double[][] result = new double[x.length][y.length];
    	
    	for (int i=0; i<x.length; i++){
    		for (int j=0; j<y.length; j++){
    			result[i][j] = x[i] * y[j];
    		}
    	}
    	return result;
    }

    // vector-matrix multiplication (y = x^T A)
    public static double[] multiply(double[] x, double[][] A) {
        int m = A.length;
        int n = A[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += A[i][j] * x[i];
        return y;
    }

    // test client
    public static void main(String[] args) {
        StdOut.println("D");
        StdOut.println("--------------------");
        double[][] d = { { 1, 2, 3 }, { 4, 5, 6 }, { 9, 1, 3} };
        StdArrayIO.print(d);
        StdOut.println();

        StdOut.println("I");
        StdOut.println("--------------------");
        double[][] c = Matrix.identity(5);
        StdArrayIO.print(c);
        StdOut.println();

        StdOut.println("A");
        StdOut.println("--------------------");
        double[][] a = Matrix.random(5, 5);
        StdArrayIO.print(a);
        StdOut.println();

        StdOut.println("A^T");
        StdOut.println("--------------------");
        double[][] b = Matrix.transpose(a);
        StdArrayIO.print(b);
        StdOut.println();

        StdOut.println("A + A^T");
        StdOut.println("--------------------");
        double[][] e = Matrix.add(a, b);
        StdArrayIO.print(e);
        StdOut.println();

        StdOut.println("A * A^T");
        StdOut.println("--------------------");
        double[][] f = Matrix.multiply(a, b);
        StdArrayIO.print(f);
        StdOut.println();
    }
}