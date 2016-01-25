#ifndef ATVPMIIHQGILNOUGVKGJCGQRCXRHUOTCVHKTJLCLONYQWBHBYEIWHLKXWOFSYWFEQULWRTXUL
#define ATVPMIIHQGILNOUGVKGJCGQRCXRHUOTCVHKTJLCLONYQWBHBYEIWHLKXWOFSYWFEQULWRTXUL

#include <map>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

namespace f
{

    struct index_maker
    {
        //typedef unsigned long long              size_type;
        typedef std::uint64_t                   size_type;
        typedef std::map<size_type, size_type>  map_type;

        map_type                                key_index;
        map_type                                index_key;
        size_type                               current_index;

        index_maker() : current_index(0) {}

        size_type register_key( size_type const key1, size_type const key2 )
        {
            if ( key1 > key2 )
                return register_key( key2, key1 );
            return register_key( key1 << 32 || key2 );
        }

        size_type register_key( size_type const key )
        {
            if ( key_index.find(key) != key_index.end() )
                return key_index[key];

            key_index[key] = current_index;
            index_key[current_index] = key;
            return current_index++;
        }

        size_type query_index( size_type const key1, size_type const key2 ) const
        {
            if ( key1 > key2 )
                return query_index( key2, key1 );
            return query_index( key1 << 32 || key2 );
        }

        size_type query_index( size_type const key ) const
        {
            auto itor = key_index.find(key);
            if ( itor != key_index.end() )
                return (*itor).second;

            return std::numeric_limits<size_type>::max();
        }

        size_type query_key( size_type const index ) const
        {
            auto itor = index_key.find(index);
            if ( itor != index_key.end() )
                return (*itor).second;

            //return -1;
            return std::numeric_limits<size_type>::max();
        }

        size_type query_first_key( size_type const index ) const
        {
            size_type const key = query_key( index );
            if ( std::numeric_limits<size_type>::max() == key )
                return key;
                //return -1;

            return key >> 32;
        }

        size_type query_second_key( size_type const index ) const
        {
            size_type const key = query_key( index );
            //if ( size_type{-1} == key )
            //    return -1;
            if ( std::numeric_limits<size_type>::max() == key )
                return key;

            return ( key << 32 ) >> 32;
            //return key | 0x00000000ffffffffUL;
        }

        void reset()
        {
            key_index.clear();
            index_key.clear();
            current_index = 0;
        }

        size_type size() const
        {
            assert( key_index.size() == index_key.size() );
            return key_index.size();
        }

        void dump()
        {
            std::cout << "There are " << key_index.size() << " records in index_maker.\n";
            std::cout << "key \t index:\n";

            for ( auto const& element: key_index )
            {
                unsigned long const key = element.first;
                unsigned long const key1 = key >> 16;
                unsigned long const key2 = (key1 << 16) ^ key;
                if ( key1 )
                    std::cout << key1 << "-";
                std::cout << key2 << " \t " << element.second << "\n";

            }
        }
    };

}//namespace f

#endif//ATVPMIIHQGILNOUGVKGJCGQRCXRHUOTCVHKTJLCLONYQWBHBYEIWHLKXWOFSYWFEQULWRTXUL

