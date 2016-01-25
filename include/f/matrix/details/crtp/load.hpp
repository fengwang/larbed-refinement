#ifndef MLOAD_HPP_INCLUDED_SDPONASD329P8HASFLKDJH498YHSDAKH39H8SFDIUH439HF39HFDF
#define MLOAD_HPP_INCLUDED_SDPONASD329P8HASFLKDJH498YHSDAKH39H8SFDIUH439HF39HFDF

#include <f/matrix/details/crtp/typedef.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_load
    {
        typedef Matrix                                                          zen_type;
        typedef typename crtp_typedef<Type, Allocator>::value_type     value_type;
        typedef typename crtp_typedef<Type, Allocator>::size_type      size_type;

        bool load( const char* const file_name )
        {
            /*
             * TODO:
             * 1) trim right of file name
             * 2) if file name with '.mat' extension
             *        call load_mat
             * 3) else
             *        call load_ascii
             */
            return load_ascii( file_name );
        }

        bool load( const std::string& file_name )
        {
            return load( file_name.c_str() );
        }

        bool load_from( const char* const file_name )
        {
            return load( file_name );
        }

        bool load_from( const std::string& file_name )
        {
            return load( file_name.c_str() );
        }

        bool load_ascii( const char* const file_name )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            std::ifstream ifs( file_name,  std::ios::in | std::ios::binary );
            if ( !ifs )
            {
                std::cerr << "Error: Failed to open file \"" << file_name << "\"\n";
                return false;
            }
            //read the file content into a string stream
            std::stringstream iss;
            std::copy( std::istreambuf_iterator<char>( ifs ), std::istreambuf_iterator<char>(), std::ostreambuf_iterator<char>( iss ) );
            const std::string& stream_buff = iss.str();
            size_type const r = std::count( stream_buff.begin(), stream_buff.end(), '\n' );
            size_type const c = std::count( stream_buff.begin(), std::find( stream_buff.begin(), stream_buff.end(), '\n' ), '\t' );
            size_type const total_elements = r * c;
            std::vector<value_type> buff;
            buff.reserve( total_elements );
            std::copy( std::istream_iterator<value_type>( iss ), std::istream_iterator<value_type>(), std::back_inserter( buff ) );
            //std::copy( std::istreambuf_iterator<value_type>( iss ), std::istreambuf_iterator<value_type>(), std::back_inserter( buff ) );
            if ( buff.size() != total_elements )
            {
                std::cerr << "Error: Failed to match matrix size.\n \tthe size of matrix stored in file \"" << file_name << "\" is " << buff.size() << ".\n";
                std::cerr << " \tthe size of the destination matrix is " << total_elements << ".\n";
                return false;
            }
            zen.resize( r, c );
            std::copy( buff.begin(), buff.end(), zen.begin() );
            ifs.close();
            return true;
        }

        //TODO: read matlab file format and impl here
        bool load_mat( const char* const file_name )
        {
            return true;
        }

    };//struct crtp_load

}

#endif//_LOAD_HPP_INCLUDED_SDPONASD329P8HASFLKDJH498YHSDAKH39H8SFDIUH439HF39HFDF

