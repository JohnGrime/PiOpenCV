/*
	Misc. utility stuff.

	Author: John Grime, The University of Oklahoma.
*/

#if !defined( UTIL_DEFINED )
#define UTIL_DEFINED

#include <cstdlib>
#include <climits>

#include <string.h>

#include <string>
#include <vector>
#include <map>

namespace Util
{

//
// As a class, to avoid unused method warnings from static "free" routines in a namespace.
//

class String
{
	public:

    //
    // Convert a character sequence into an integer or floating point type
    //
	template< typename T >
	static bool ToInteger( const char *str, T& result, int base = 10 )
	{
		if( str == nullptr || base < 2 ) return false;

		char *endptr;
		result = strtoll( str, &endptr, base );
		if( (endptr==str) || (*endptr!='\0') ) return false;
		return true;
	}
	template< typename T >
	static bool ToInteger( const std::string& str, T& result, int base = 10 )
	{
		return ToInteger( str.c_str(), result, base );
	}

	template< typename T >
	static bool ToReal( const char *str, T& result )
	{
		if( str == nullptr ) return false;

		char *endptr;
		result = strtod( str, &endptr );
		if( (endptr==str) || (*endptr!='\0') ) return false;
		return true;
	}
	template< typename T >
	static bool ToReal( const std::string& str, T& result )
	{
		return ToReal( str.c_str(), result );
	}

	template< typename T >
	static bool ToNumber( const char *str, T& result, int base = 10 )
	{
		if( std::is_integral<T>::value ) return ToInteger( str, result, base );
		else if( std::is_floating_point<T>::value ) return ToReal( str, result );
		return false;
	}
	template< typename T >
	static bool ToNumber( const std::string& str, T& result, int base = 10 )
	{
		return ToNumber( str.c_str(), result, base );
	}

	//
	// This is slow, as we repeatedly append to an std::string, and then copy it into "results". This approach
	// does mean we don't need any nasty fixed-size intermediate buffers etc, so it's hopefully safer.
	//
	static int Tokenize( const char * source, std::vector< std::string > &results, const char *delimiters, bool allow_empty = false )
	{
		std::string temp;

		results.clear(); // BEFORE error return test ...

		if( source == nullptr || delimiters == nullptr ) return 0;

		size_t src_len = strlen( source );

		for( size_t i=0; i<src_len; i++ )
		{
			if( is_in( source[i], delimiters ) == true )
			{
				if( allow_empty || (temp.size()>0) ) results.push_back( temp );
				temp.clear();
			}
			else
			{
				temp += source[i];
			}
		}
		// If there's a trailing token, add to the results vector
		if( temp.size() > 0  ) results.push_back( temp );

		return (int)results.size();
	}
	static int Tokenize( const std::string& source, std::vector< std::string > &results, const char *delimiters, bool allow_empty = false )
	{
		return Tokenize( source.c_str(), results, delimiters, allow_empty );
	}
    
	static int Tokenize( const char *source, std::vector<std::string> &results, const char *word_delimiters, const char *string_delimiters, bool allow_empty = false )
	{
		const char* add_until = nullptr; // if we "open" quoted token, add chars until this "close" character
		std::string str;

		results.clear(); // BEFORE error return test ...

		// allow string_delimiters == nullptr, but we need valid source string and word_delimiters
		if( (source==nullptr) || (word_delimiters==nullptr) ) return 0;

		for (size_t i=0, max_i=strlen(source); i<max_i; i++)
		{
			bool push_token = false;
			auto c = source[i];

			if (add_until!=nullptr) // in string?
			{
				if (*add_until==c) // string closed?
				{
					add_until = nullptr;
					push_token = true;
				}
				else str += c;
			}
			else if (is_in(c,string_delimiters)) add_until = &source[i]; // NOT "&c": c temporary
			else if (is_in(c,word_delimiters)) push_token = true;
			else str += c;

			if (push_token && ((str.size()>0)||allow_empty))
			{
				results.push_back(str);
				str = "";
			}
		}

		if (str.size()>0) results.push_back(str);

		return (int)results.size();
	}

	private:

	//
	// Check if character 'test' is in test_characters (assumes "test_characters" null-terminated!).
	//
	static bool is_in( char test, const char * test_characters )
	{
		if( test_characters == nullptr ) return false;

		while( *test_characters != '\0' )
		{
			if( test == *(test_characters++) ) return true;
		}
		return false;
	}

	//
	// Check if "check" starts with string "str"; used in variable expansion routines.
	//
	static bool starts_with( const char *str, const char *check, int check_len )
	{
		if( str[0] == '\0' || check[0] == '\0' ) return false;

		for( int i=0; i<check_len && str[i] != '\0'; i++ )
		{
			if( str[i] != check[i] ) return false;
		}
		return true;
	}
};

//
// Running statistics, based on algorithms of B. P. Welford
// (via Knuth, "The Art of Computer Programming").
//
// This algorithm provides not only the capability to determine
// variance (and hence, stdev and stderr) from a running input
// with very little storage, it is also robust to catastrophic
// cancellation.
//
struct Stats
{
	size_t N; // number of samples so far
	double S; // N*sigma^2
	double min, mean, max;
	
	Stats()
	{
		Clear();
	}
	void Clear()
	{
		N = 0;
		S = min = mean = max = 0.0;
	}
	void AddSample( double x )
	{
		N++;
		if( N == 1 )
		{
			min = mean = max = x;
			S = 0.0;
			return;
		}
		
		//
		// Update values (new marked with prime):
		//   mean' = mean + (x-mean)/N
		//   S' = S + (x-mean)*(x-mean')
		//

		double delta = (x-mean);
		mean += delta/N;
		S += delta * (x-mean); // <- note: uses updated value of "mean" as wella s old value via "delta".

		if( x < min ) min = x;
		if( x > max ) max = x;
	}
	double Sum() const
	{
		return mean*N;
	}
	double Variance() const
	{
		// SAMPLE variance
		return (N>1) ? (S/(N-1)) : (0.0);
	}
	double StdDev() const
	{
		// SAMPLE standard deviation
		return sqrt( Variance() );
	}
	double StdErr() const
	{
		// Estimated standard error of the sample mean:
		// SE = stdev / sqrt(N) : stdev = sample standard deviation
		// SE = sqrt(variance) / sqrt(N) : as stdev = sqrt(variance)
		// SE = sqrt( variance / N ) : as sqrt() distributive
		return (N>1) ? ( sqrt(Variance()/N) ) : (0.0);
	}
	
	// Combine separate sets of sample stats
	Stats& operator += ( const Stats &rhs )
	{
		// Temporary values, in case &rhs == this
		size_t new_N;
		double new_sum, new_mean, new_S;
		
		// Ignore if no data present in rhs
		if( rhs.N < 1 ) return *this;
				
		new_N = N + rhs.N;
		new_sum = (mean*N) + (rhs.mean*rhs.N);
		new_mean = new_sum / new_N; // safe: rhs.N >= 1, so new_N >= 1.
		
		// This is basically the "parallel algorithm" version of the "online" algorithm
		// for calculating variance in one pass when the sample is partitioned into multiple
		// sets. This is attributed to Chan et al, "Updating Formulae and a Pairwise Algorithm for
		// Computing Sample Variances.", Technical Report STAN-CS-79-773, Stanford CS, (1979).
		//
		// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		//
		double delta = (mean - rhs.mean);
		new_S = (S + rhs.S) + (delta*delta)*(N*rhs.N)/new_N;
		
		// If no samples in this Stats structure, or if rhs min/max should replace current:
		if( N < 1 || rhs.min < min ) min = rhs.min;
		if( N < 1 || rhs.max > max ) max = rhs.max;
		
		// Only change N now, as we needed it in the min/max update above.
		N = new_N;
		S = new_S;
		mean = new_mean;
		
		return *this;
	}
};

//
// Sets of statistics, accessible using strin keys or integer indices.
//

struct StatsSet
{
	std::map<std::string, int> key_to_idx;
	std::vector<Stats> stats_vec;

	int AddName( const char* key )
	{
		const auto& it = key_to_idx.find(key);
		if (it != key_to_idx.end()) return it->second;
		
		Stats s;
		int idx = (int)stats_vec.size();
		key_to_idx[key] = idx;
		stats_vec.push_back(s);

		return idx;
	}

	int AddSampleByName( const char* key, double val )
	{
		const auto& it = key_to_idx.find(key);
		int idx = (it!=key_to_idx.end()) ? (it->second) : (AddName(key));
		stats_vec[idx].AddSample(val);
		return idx;
	}

	int AddSampleByIndex( int idx, double val )
	{
		stats_vec[idx].AddSample(val);
		return idx;
	}

	void Clear()
	{
		for (auto& s : stats_vec) s.Clear();
	}
};

}

#endif
