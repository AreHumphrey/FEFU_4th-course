using System;

public delegate bool MyDelegate(Action<float> processor, int count, float value);

class Program
{

    static bool ProcessWithLogging(Action<float> action, int count, float value)
    {
        Console.WriteLine($"[Метод 1] Запуск: count={count}, value={value}");
        for (int i = 0; i < count; i++)
        {
            action(value + i * 0.5f);
        }
        return value > 0 && count > 0;
    }


    static bool ProcessIfValid(Action<float> action, int count, float value)
    {
        if (count >= 2 && value >= 5.0f)
        {
            Console.WriteLine($"[Метод 2] Условие выполнено. Выполняем действие...");
            action(value * count);
            return true;
        }
        Console.WriteLine($"[Метод 2] Условие НЕ выполнено.");
        return false;
    }

    static void PrintValue(float x)
    {
        Console.WriteLine($"  ➤ Обработка значения: {x:F2}");
    }

    static void Main(string[] args)
    {
        Console.WriteLine("Лабораторная работа №1: Делегаты в C# \n");

        MyDelegate del1 = ProcessWithLogging;
        MyDelegate del2 = ProcessIfValid;

        Action<float> printer = PrintValue;

        Console.WriteLine("→ Вызов метода 1:");
        bool res1 = del1(printer, 3, 2.0f);
        Console.WriteLine($"Результат: {res1}\n");

        Console.WriteLine("→ Вызов метода 2 (неудачный):");
        bool res2 = del2(printer, 1, 3.0f);
        Console.WriteLine($"Результат: {res2}\n");

        Console.WriteLine("→ Вызов метода 2 (успешный):");
        bool res3 = del2(printer, 4, 6.0f);
        Console.WriteLine($"Результат: {res3}\n");

        MyDelegate lambdaDel = (act, c, v) =>
        {
            Console.WriteLine($"[Лямбда] c={c}, v={v}");
            act(v + c);
            return v + c > 10;
        };

        Console.WriteLine("→ Вызов через лямбду:");
        bool res4 = lambdaDel(printer, 5, 6.5f);
        Console.WriteLine($"Результат лямбды: {res4}");
    }
}
