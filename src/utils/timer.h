#pragma once

#include <iostream>
#include <chrono>

class Timer
{
public:
	Timer(const char* name)
		: m_Name(name), m_StartTimepoint(std::chrono::high_resolution_clock::now()) { }
	
	~Timer()
	{
		auto endTimepoint = std::chrono::high_resolution_clock::now();
		auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTimepoint - m_StartTimepoint).count();
		double ms = us * 0.001;
		std::cout << m_Name << " duration = " << ms << "ms (" << us << "us)\n";
	}

private:
	const char* m_Name;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
};