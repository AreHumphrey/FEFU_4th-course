using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ParallelLabs.Labs.Lab04
{
    public static class Lab04Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №4: Обратные асинхронные вызовы (через Task + callback)\n");

            int rows = 4;
            int cols = 5;

            Console.WriteLine($"Генерация матрицы {rows}x{cols} и фильтрация чётных чисел...\n");

            Console.WriteLine("1. Использование метода как обратного вызова:");
            StartAsyncWithCallback(rows, cols, EvenNumbersCallback);

            for (int i = 0; i < 10; i++)
            {
                Console.Write(".");
                Thread.Sleep(100);
            }
            Console.WriteLine("\nГлавный поток завершил свою работу.\n");

            Thread.Sleep(2500);

            Console.WriteLine("2. Использование лямбда-выражения как обратного вызова:");
            StartAsyncWithCallback(rows, cols, (result) =>
            {
                Console.WriteLine($"\n Лямбда-колбэк: найдено {result.Count} чётных чисел.");
                Console.WriteLine("Чётные числа: " + string.Join(", ", result));
            });

            Thread.Sleep(2500);

        }

        private static void StartAsyncWithCallback(int rows, int cols, Action<List<int>> callback)
        {
            _ = Task.Run(() =>
            {
                var result = GenerateMatrixAndFilterEven(rows, cols);
                callback(result); 
            });
        }

        private static List<int> GenerateMatrixAndFilterEven(int rows, int cols)
        {
            var evenNumbers = new List<int>();
            var rand = new Random();

            Console.WriteLine($"[Фон] Генерация матрицы {rows}x{cols}...");

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int num = rand.Next(1, 101); 
                    if (num % 2 == 0)
                    {
                        evenNumbers.Add(num);
                    }
                }
            }

            Thread.Sleep(2000); 
            return evenNumbers;
        }


        private static void EvenNumbersCallback(List<int> result)
        {
            Console.WriteLine($"\n Метод-колбэк: найдено {result.Count} чётных чисел.");
            Console.WriteLine("Чётные числа: " + string.Join(", ", result));
        }
    }
}