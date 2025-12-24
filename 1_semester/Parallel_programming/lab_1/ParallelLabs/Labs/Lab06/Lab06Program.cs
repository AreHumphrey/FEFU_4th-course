using System;
using System.Threading;

// Передать параметры в поток двумя способами
// Найти скалярное произведение двух случайных векторов

namespace ParallelLabs.Labs.Lab06
{
    public class VectorData
    {
        public int Size;
        public int[] VectorA; 
        public int[] VectorB; 
        public int Result;
    }

    public static class Lab06Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №6: Передача данных потокам\n");
            int vectorSize = 5;
            Console.WriteLine($"Вычисление скалярного произведения двух векторов размером {vectorSize}...\n");

            Console.WriteLine("1. Передача параметров через ParameterizedThreadStart:");
            var data1 = new VectorData { Size = vectorSize };
            var thread1 = new Thread(new ParameterizedThreadStart(CalculateDotProduct));
            thread1.Start(data1);
            thread1.Join();


            Console.WriteLine($"   Вектор A: [{string.Join(", ", data1.VectorA)}]");
            Console.WriteLine($"   Вектор B: [{string.Join(", ", data1.VectorB)}]");
            Console.WriteLine($"   Скалярное произведение: {data1.Result}\n");

            Console.WriteLine("2. Передача параметров через экземпляр класса:");
            var worker = new DotProductWorker(vectorSize);
            var thread2 = new Thread(worker.Calculate);
            thread2.Start();
            thread2.Join();


            Console.WriteLine($"   Вектор A: [{string.Join(", ", worker.VectorA)}]");
            Console.WriteLine($"   Вектор B: [{string.Join(", ", worker.VectorB)}]");
            Console.WriteLine($"   Скалярное произведение: {worker.Result}\n");

            Console.WriteLine("Лабораторная работа №6 успешно выполнена.");
        }

        private static void CalculateDotProduct(object? obj)
        {
            if (obj is not VectorData data)
                return;

            var rand = new Random();
            data.VectorA = new int[data.Size];
            data.VectorB = new int[data.Size];


            for (int i = 0; i < data.Size; i++)
            {
                data.VectorA[i] = rand.Next(1, 11);
                data.VectorB[i] = rand.Next(1, 11);
            }


            int dotProduct = 0;
            for (int i = 0; i < data.Size; i++)
            {
                dotProduct += data.VectorA[i] * data.VectorB[i];
            }

            data.Result = dotProduct;

            Console.WriteLine($"   [Поток {Thread.CurrentThread.ManagedThreadId}] Скалярное произведение вычислено.");
        }
    }

    public class DotProductWorker
    {
        public int Size { get; }
        public int[] VectorA { get; private set; } 
        public int[] VectorB { get; private set; } 
        public int Result { get; private set; }

        public DotProductWorker(int size)
        {
            Size = size;
            VectorA = new int[size];
            VectorB = new int[size];
        }

        public void Calculate()
        {
            var rand = new Random();


            for (int i = 0; i < Size; i++)
            {
                VectorA[i] = rand.Next(1, 11);
                VectorB[i] = rand.Next(1, 11);
            }


            Result = 0;
            for (int i = 0; i < Size; i++)
            {
                Result += VectorA[i] * VectorB[i];
            }

            Console.WriteLine($"   [Поток {Thread.CurrentThread.ManagedThreadId}] Скалярное произведение вычислено.");
        }
    }
}