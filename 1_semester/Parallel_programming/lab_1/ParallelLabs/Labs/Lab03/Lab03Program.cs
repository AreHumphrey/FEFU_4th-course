using System;
using System.Threading;
using System.Threading.Tasks;

namespace ParallelLabs.Labs.Lab03
{
    public static class Lab03Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №3: Ожидание завершения асинхронного метода с тайм-аутом\n");

            int rows = 5;
            int cols = 5;
            int timeoutMs = 50; 

            Console.WriteLine($"Генерация матрицы {rows}x{cols} с тайм-аутом {timeoutMs} мс...\n");

            // Запускаем операцию как Task
            var task = Task.Run(() => GenerateRandomBitMatrix(rows, cols));

            Console.Write("Ожидание завершения... ");


            while (!task.IsCompleted)
            {
                Console.Write(".");
                Thread.Sleep(timeoutMs);
            }

            Console.WriteLine("\n Операция завершена)");

            int[,] matrix = task.Result;

            Console.WriteLine("\nСгенерированная матрица:");
            PrintMatrix(matrix);
        }

        private static int[,] GenerateRandomBitMatrix(int rows, int cols)
        {
   
            Thread.Sleep(2000);

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