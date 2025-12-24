using System;
using System.Threading;

namespace ParallelLabs.Labs.Lab05
{
    public static class Lab05Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №5: Применение класса Thread\n");

            int threadCount = 3; // по заданию — можно менять
            int rows = 5;
            int cols = 5;

            // Массив потоков
            Thread[] threads = new Thread[threadCount];

            Console.WriteLine($"Создано {threadCount} потоков для подсчёта чётных элементов в матрицах {rows}x{cols}.\n");

            for (int i = 0; i < threadCount; i++)
            {
                int threadId = i + 1; // чтобы нумерация была с 1

                threads[i] = new Thread(() =>
                {
                    var matrix = GenerateRandomMatrix(rows, cols);
                    int evenCount = CountEvenElements(matrix);

                    Console.WriteLine($"\n✅ Поток {threadId}: Завершил обработку.");
                    Console.WriteLine($"   Матрица {rows}x{cols}:");
                    PrintMatrix(matrix);
                    Console.WriteLine($"   Чётных элементов: {evenCount}");
                });

                threads[i].Start();
                Console.WriteLine($"Поток {threadId} запущен. Id: {threads[i].ManagedThreadId}");
            }

            Console.WriteLine("\nОжидание завершения всех потоков...");

            // Мониторинг: ждём завершения всех потоков
            foreach (Thread t in threads)
            {
                t.Join(); // блокируем главный поток до завершения каждого
            }

            Console.WriteLine("\n✅ Все потоки завершили работу.");
            Console.WriteLine("Лабораторная работа №5 успешно выполнена.");
        }

        // Генерация случайной матрицы
        private static int[,] GenerateRandomMatrix(int rows, int cols)
        {
            var matrix = new int[rows, cols];
            var rand = new Random();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = rand.Next(1, 101); // от 1 до 100
                }
            }

            return matrix;
        }

        // Подсчёт чётных элементов
        private static int CountEvenElements(int[,] matrix)
        {
            int count = 0;
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (matrix[i, j] % 2 == 0)
                    {
                        count++;
                    }
                }
            }

            return count;
        }

        // Вывод матрицы
        private static void PrintMatrix(int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Console.Write($"{matrix[i, j],3} ");
                }
                Console.WriteLine();
            }
        }
    }
}