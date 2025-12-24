using System;
using System.Collections.Generic;
using System.Threading;

// Продемонстрировать фоновый и приоритетный потоки
//  Статистика по символам в строке (символ → количество вхождений)
namespace ParallelLabs.Labs.Lab07
{

    public class StringAnalyzer
    {
        private readonly string _text;

        public StringAnalyzer(string text)
        {
            _text = text ?? throw new ArgumentNullException(nameof(text));
        }


        public Dictionary<char, int> Analyze()
        {
            var stats = new Dictionary<char, int>();

            Console.WriteLine($"[Поток {Thread.CurrentThread.ManagedThreadId}] Начинаю анализ строки...");

            for (int i = 0; i < _text.Length; i++)
            {
                char c = _text[i];
                if (stats.ContainsKey(c))
                    stats[c]++;
                else
                    stats[c] = 1;

   
                Thread.Sleep(50); 
            }

            Console.WriteLine($"[Поток {Thread.CurrentThread.ManagedThreadId}] Анализ завершён.");
            return stats;
        }
    }

    public static class Lab07Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №7: Создание и управление потоками\n");

 
            string input = "Hello, World! This is a test string for thread pool analysis.";

            Console.WriteLine($"Анализируем строку: \"{input}\"\n");


            Console.WriteLine("1. Асинхронный анализ через пул потоков (фоновый поток):");
            var analyzer = new StringAnalyzer(input);

            // создаем фоновый поток
            ThreadPool.QueueUserWorkItem(_ =>
            {
                var result = analyzer.Analyze();
                PrintStats(result);
            });

 
            for (int i = 0; i < 10; i++)
            {
                Console.Write(".");
                Thread.Sleep(200);
            }
            Console.WriteLine("\nГлавный поток завершил свою работу.\n");

    
            Thread.Sleep(3000);

            // создаем приоретентный поток
            Console.WriteLine("2. Асинхронный анализ через приоритетный поток:");
            var thread = new Thread(() =>
            {
                var analyzer2 = new StringAnalyzer(input);
                var result = analyzer2.Analyze();
                PrintStats(result);
            })
            {
                IsBackground = false,
                Name = "PriorityAnalyzer"
            };

            thread.Start();

 
            thread.Join();

            Console.WriteLine("\n Лабораторная работа №7 успешно выполнена.");
        }


        private static void PrintStats(Dictionary<char, int> stats)
        {
            Console.WriteLine("\n Статистика по символам:");
            foreach (var kvp in stats)
            {
                Console.WriteLine($"   '{kvp.Key}' — встречается {kvp.Value} раз{(kvp.Value > 1 ? "а" : "")}");
            }
        }
    }
}