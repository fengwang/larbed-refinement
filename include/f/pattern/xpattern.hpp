#ifndef BDRFUEILKTDHSJSOGOVHBCAOWYBXWBMDWYQNXKSNXNOFWNASGBQGUTNXGRNGFXRWUPRNMEONI
#define BDRFUEILKTDHSJSOGOVHBCAOWYBXWBMDWYQNXKSNXNOFWNASGBQGUTNXGRNGFXRWUPRNMEONI

#include <f/matrix/matrix.hpp>
#include <f/lexical_cast/lexical_cast.hpp>
#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/variate_generator/variate_generator.hpp>
#include <f/coefficient/coefficient.hpp>

#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <cstddef>
#include <string>
#include <complex>
#include <map>

namespace f
{
    template< typename T >
    struct xpattern
    {
        typedef T                               value_type;
        typedef std::complex<T>                 complex_type;
        typedef matrix<value_type>              matrix_type;
        typedef std::vector<matrix_type>        matrix_vector_type;
        typedef matrix<complex_type>            complex_matrix_type;
        typedef std::size_t                     size_type;
        typedef matrix<size_type>               size_matrix_type;
        typedef std::vector<size_matrix_type>   size_matrix_vector_type;
        typedef std::vector<size_type>          size_vector_type;
        typedef std::vector<value_type>         array_type;

        //-------------------------------------------------------------------------------
        complex_type                            thickness;      // only for initial thickness
        size_type                               ug_size;
        size_type                               tilt_size;
        size_type                               column_index;
        size_matrix_vector_type                 ar;             // ar[][][]
        matrix_vector_type                      diag;           // diag[][][]
        matrix_vector_type                      intensity;      // intensity[][][]
        complex_matrix_type                     ug;             //
        array_type                              thickness_array;// vary thickness for every tilt

        void update_thickness( value_type val )
        {
            thickness = complex_type{ 0.0, val };
            thickness_array.resize( tilt_size );
            std::fill( thickness_array.begin(), thickness_array.end(), val );
        }

        void add_noise( T const level = 0.01 )
        {
            variate_generator<T> vg( T{-1.0}, T{1.0} );
            for ( auto& mat : intensity )
                std::for_each( mat.begin(), mat.end(), [&vg,level]( value_type& x ) { x += vg() * level; });
        }

        void renormalize_intensities()
        {
            for ( auto&& it : intensity )
            {
                std::for_each( it.begin(), it.end(), [](value_type& x){ if ( x < value_type{0} ) x = value_type{0}; } );
                value_type sum_up = std::accumulate( it.begin(), it.end(), value_type{0});
                std::for_each( it.begin(), it.end(), [sum_up](value_type& x){ x /= sum_up; } );
            }
        }

        void normalize_intensities()
        {
            for ( auto&& it : intensity )
            {
                std::for_each( it.begin(), it.end(), [](value_type& x){ if ( x < value_type{0} ) x = value_type{0}; } );
                value_type sum_up = std::accumulate( it.begin(), it.end(), value_type{0});
                std::for_each( it.begin(), it.end(), [sum_up](value_type& x){ x /= sum_up; } );
            }
        }

        friend std::ostream& operator << ( std::ostream& os, xpattern<T> const& pt )
        {
            os << "thickness:\t"  << pt.thickness << "\n";
            os << "ug_size:\t"  << pt.ug_size << "\n";
            os << "tilt_size:\t"  << pt.tilt_size << "\n";
            os << "column_index:\t"  << pt.column_index << "\n";
            os << "thickness array is \n";
            std::copy( pt.thickness_array.begin(), pt.thickness_array.end(), std::ostream_iterator<T>( os, "\t" ) );
            os << "\n";
#if 0
            os << "ar:\n"  << pt.ar[0] << "\n";

            os << "diag\n";
            for ( std::size_t i = 0; i != pt.tilt_size; ++i )
                os << pt.diag[i].transpose() << "\n";

            os << "intensity\n";
            for ( std::size_t i = 0; i != pt.tilt_size; ++i )
                os << pt.intensity[i].transpose() << "\n";

            os << "ug:\n"  << pt.ug << "\n";
#endif
            return os;
        }

    };//struct xpattern

    static bool is_file_exist( std::string const& file_path )
    {
        std::ifstream ifs( file_path );
        return ifs.good();
    }

    template< typename T >
    xpattern<T> const make_xpattern( std::string const& dir_path_, std::complex<T> const& thickness_, std::size_t column_index_ = 0 )
    {
        typedef std::size_t size_type;
        typedef T value_type;
        typedef matrix<value_type> matrix_type;
        typedef matrix<size_type> size_matrix_type;
        typedef std::complex<value_type> complex_type;
        xpattern<T> pt;
        pt.column_index = column_index_;

        if ( std::abs( std::real(thickness_) ) > T{1.0e-10} ) return pt;
        if ( std::imag(thickness_)  < T{1.0e-10} ) return pt;

        pt.thickness = thickness_;

        //load ug
        std::string const& ug_file_path = dir_path_ + std::string{"/_UgMasterList.txt"};
        std::ifstream ifs_ug{ ug_file_path };
        if ( !ifs_ug.good() ) return pt;

        std::stringstream iss;
        std::copy( std::istreambuf_iterator<char>{ifs_ug}, std::istreambuf_iterator<char>{}, std::ostreambuf_iterator<char>{iss} );
        std::string orig_ug_str = iss.str();
        std::for_each( orig_ug_str.begin(), orig_ug_str.end(), [](char& ch){ if (':' == ch) ch = ' ';} );
        iss.str(orig_ug_str);

        std::vector<T> buff;
        std::copy( std::istream_iterator<T>(iss), std::istream_iterator<T>(), std::back_inserter(buff) );
        matrix<T> ug_tmp{ buff.size(), 1};
        ug_tmp.reshape( buff.size()/3, 3 );
        std::copy( buff.begin(), buff.end(), ug_tmp.begin() );

        pt.ug.resize( ug_tmp.row(), 1 );
        for ( size_type r = 0; r != ug_tmp.row(); ++r )
            pt.ug[r][0] = complex_type{ ug_tmp[r][1], ug_tmp[r][2] };

        size_matrix_type ar_x;
        matrix_type diag_x;
        matrix_type intensity_x;
        for ( size_type index = 0; true; ++index )
        {
            std::string const& id = std::to_string( index );
            std::string const& ar_file_path = dir_path_ + std::string{"/Ar_"} + id + std::string{".txt"};
            std::string const& diag_file_path = dir_path_ + std::string{"/Diag_"} + id + std::string{".txt"};
            std::string const& intensity_file_path = dir_path_ + std::string{"/Intensities_"} + id + std::string{".txt"};

            if ( ! ( is_file_exist( ar_file_path ) && is_file_exist( diag_file_path ) && is_file_exist( intensity_file_path ) ) ) break;

            ar_x.load( ar_file_path );
            diag_x.load( diag_file_path );
            intensity_x.load( intensity_file_path );

            pt.ar.push_back( ar_x );
            pt.diag.push_back( diag_x );
            pt.intensity.push_back( intensity_x );

        }

        pt.ug_size = 0;
        for ( auto&& ar_ : pt.ar )
            pt.ug_size = std::max( pt.ug_size, *std::max_element(ar_.begin(), ar_.end()) );

        pt.ug_size++;

        pt.tilt_size = pt.diag.size();

        pt.update_thickness( std::imag(thickness_) );

        return pt;
    }

    template< typename T >
    std::map<unsigned long, T> const extract_ug_norm( xpattern<T> const& pt )
    {
        typedef T                   value_type;
        typedef unsigned long       size_type;

        //       ug_index     sum weighed_ug^2     sum weighs
        std::map<size_type, std::pair<value_type, value_type> > record;
        value_type const threshold = 0.1;
        value_type const factor = 12.56637061435917295384;
        value_type const small_factor = threshold * 1.0e-10;

        for ( size_type index = 0; index != pt.tilt_size; ++index )
        {
            //auto coef = coefficient<value_type>{ pt.thickness, (pt.diag)[index].begin(), (pt.diag)[index].end() };
            auto coef = coefficient<value_type>{ std::complex<T>{0, pt.thickness_array[index]}, (pt.diag)[index].begin(), (pt.diag)[index].end() };
            for ( size_type jndex = 0; jndex != (pt.diag)[index].size(); ++jndex )
            {
                if ( jndex == pt.column_index ) continue;

                size_type const row = jndex;
                size_type const col = pt.column_index;
                value_type const c1_norm = std::norm( coef( row, col ) );  // |C_ij|^2
                value_type const I_ij = (pt.intensity)[index][row][0];
                size_type const ug_index = (pt.ar)[index][row][col];
                value_type const fitted = std::sqrt( I_ij / c1_norm );
                if ( fitted > threshold ) continue;
                value_type const weight = std::exp( 1+ factor * I_ij / (fitted+small_factor) );
                value_type const weighted = fitted * weight;

                if ( record.find( ug_index ) == record.end() )
                {
                    record[ug_index] = std::make_pair( weighted, weight );
                    continue;
                }

                record[ug_index].first += weighted;
                record[ug_index].second += weight;
            }
        }

        std::map<size_type, value_type> ug_norm;

        for ( auto const& elem : record )
        {
            size_type const ug_index = elem.first;
            value_type const sum_weighted_ug = elem.second.first;
            value_type const sum_weigh = elem.second.second;
            value_type const the_norm = sum_weighted_ug / sum_weigh;
            ug_norm[ug_index] = the_norm;
        }

        return ug_norm;
    }





}//namespace f

#endif//BDRFUEILKTDHSJSOGOVHBCAOWYBXWBMDWYQNXKSNXNOFWNASGBQGUTNXGRNGFXRWUPRNMEONI

