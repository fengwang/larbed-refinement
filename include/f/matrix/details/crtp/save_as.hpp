#ifndef MSAVE_AS_HPP_INCLUDED_SDPOJNAS23O9UHAFSLKJAHF9YH4AKJFHASKLFJH39HUAFKJHSF
#define MSAVE_AS_HPP_INCLUDED_SDPOJNAS23O9UHAFSLKJAHF9YH4AKJFHASKLFJH39HUAFKJHSF

#include <f/matrix/details/crtp/typedef.hpp>

#include <fstream>
#include <iostream>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_save_as
    {
        typedef Matrix      zen_type;

        bool save_as( const char* const file_name ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            std::ofstream ofs( file_name );
            if ( !ofs ) { return false; }
            ofs.precision( 16 );
            ofs << zen;
            ofs.close();
            return true;
        }

        bool save_as( const std::string& file_name ) const
        {
            return save_as( file_name.c_str() );
        }

        bool save_to( const char* const file_name ) const
        {
            return save_as( file_name );
        }

        bool save_to( const std::string& file_name ) const
        {
            return save_as( file_name.c_str() );
        }

        bool save( const char* const file_name ) const
        {
            return save_as( file_name );
        }

        bool save( const std::string& file_name ) const
        {
            return save_as( file_name.c_str() );
        }

    };//struct crtp_save_as

}

#endif//_SAVE_AS_HPP_INCLUDED_SDPOJNAS23O9UHAFSLKJAHF9YH4AKJFHASKLFJH39HUAFKJHSF

