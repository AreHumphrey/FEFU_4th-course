using System;
using System.Threading.Tasks;

namespace ParallelLabs.Labs.Lab08
{
    public static class Lab08Program
    {
        public static void Run()
        {
            Console.WriteLine("Лабораторная работа №8: Создание и запуск задач (класс Task)\n");

            int rows = 3;
            int cols = 4;

            Console.WriteLine($"Генерация и преобразование {rows}x{cols} матриц...\n");

            Task<double[,]>[] tasks = new Task<double[,]>[3];

            Console.WriteLine("1. Запуск через конструктор Task и Start():");
            var matrix1 = GenerateRandomMatrix(rows, cols);
            tasks[0] = new Task<double[,]>(() => TransformMatrix(matrix1));
            tasks[0].Start();
            PrintTaskInfo(tasks[0]);

            Console.WriteLine("\n2. Запуск через TaskFactory.StartNew():");
            var matrix2 = GenerateRandomMatrix(rows, cols);
            var factory = new TaskFactory();
            tasks[1] = factory.StartNew(() => TransformMatrix(matrix2));
            PrintTaskInfo(tasks[1]);

            Console.WriteLine("\n3. Запуск через Task.Factory.StartNew():");
            var matrix3 = GenerateRandomMatrix(rows, cols);
            tasks[2] = Task.Factory.StartNew(() => TransformMatrix(matrix3));
            PrintTaskInfo(tasks[2]);

            Console.WriteLine("\nОжидание завершения всех задач...");
            Task.WaitAll(tasks);

            Console.WriteLine("\nВсе задачи завершены.");
            Console.WriteLine("Лабораторная работа №8 успешно выполнена.");
        }

        private static int[,] GenerateRandomMatrix(int rows, int cols)
        {
            var matrix = new int[rows, cols];
            var rand = new Random();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = rand.Next(-10, 11); 
            return matrix;
        }

        private static double[,] TransformMatrix(int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = Math.Sin(matrix[i, j]);
            return result;
        }

        private static void PrintTaskInfo(Task<double[,]> task)
        {
            Console.WriteLine($"   Id задачи: {task.Id}");
            Console.WriteLine($"   Состояние: {task.Status}");
        }
    }
}
