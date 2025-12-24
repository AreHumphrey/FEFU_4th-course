using System;
using System.Threading.Tasks;

//  Выполнить операцию асинхронно тремя способами
// Генерация матрицы случайных битов 

namespace ParallelLabs.Labs.Lab02
{

    public delegate Task<int[,]> GenerateMatrixAsyncDelegate(int rows, int cols);

    public static class Lab02Program
    {

        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №2: Асинхронные делегаты и многопоточность\n");

            int rows = 5;
            int cols = 5;

            Console.WriteLine($"Генерация матрицы {rows}x{cols} с случайными битами (0/1)...\n");

            RunAsync(rows, cols).Wait();
        }

        private static async Task RunAsync(int rows, int cols)
        {

            Console.WriteLine("1. Использование пользовательского делегата:");
            GenerateMatrixAsyncDelegate userDel = GenerateRandomBitMatrixAsync;
            var matrix1 = await userDel(rows, cols);
            PrintMatrix(matrix1);

            Console.WriteLine("\n2. Использование библиотечного делегата Func<T>:");
            Func<int, int, Task<int[,]>> funcDel = GenerateRandomBitMatrixAsync;
            var matrix2 = await funcDel(rows, cols);
            PrintMatrix(matrix2);

            Console.WriteLine("\n3. Использование лямбда-выражения:");
            Func<int, int, Task<int[,]>> lambdaDel = async (r, c) =>
            {
                Console.WriteLine($"[Лямбда] Генерация матрицы {r}x{c}...");
                await Task.Delay(100);
                return GenerateRandomBitMatrix(r, c);
            };
            var matrix3 = await lambdaDel(rows, cols);
            PrintMatrix(matrix3);

            Console.WriteLine("\nВсе три способа успешно выполнили генерацию матрицы.");
        }

        private static int[,] GenerateRandomBitMatrix(int rows, int cols)
        {
            var matrix = new int[rows, cols];
            var rand = new Random();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = rand.Next(2);
                }
            }

            return matrix;
        }

        private static async Task<int[,]> GenerateRandomBitMatrixAsync(int rows, int cols)
        {
            Console.WriteLine($"[Метод] Генерация матрицы {rows}x{cols} началась...");
            await Task.Delay(500); 
            var matrix = GenerateRandomBitMatrix(rows, cols);
            Console.WriteLine($"[Метод] Генерация завершена.");
            return matrix;
        }

        private static void PrintMatrix(int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Console.Write($"{matrix[i, j]} ");
                }
                Console.WriteLine();
            }
        }
    }
}