using System;
using System.Threading.Tasks;

namespace ParallelLabs.Labs.Lab09
{
    public static class Lab09Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №9: Задачи продолжения\n");

            var generateTask = Task.Run(() =>
            {
                var rand = new Random();
                int size = rand.Next(5, 21);
                var arr = new int[size];
                Console.WriteLine($"[Генерация] Создан массив размером {size}");
                for (int i = 0; i < size; i++)
                    arr[i] = rand.Next(1, 101);
                return arr;
            });

            var sumOdd = generateTask.ContinueWith(t =>
            {
                int sum = 0;
                foreach (int x in t.Result)
                    if (x % 2 != 0) sum += x;
                Console.WriteLine($"[Сумма нечётных] = {sum}");
                return sum;
            });

            var countDiv3 = generateTask.ContinueWith(t =>
            {
                int count = 0;
                foreach (int x in t.Result)
                    if (x % 3 == 0) count++;
                Console.WriteLine($"[Кратных 3] = {count}");
                return count;
            });

            Task.WaitAll(sumOdd, countDiv3);
            Console.WriteLine("\n✅ Лабораторная работа №9 успешно выполнена.");
        }
    }
}
