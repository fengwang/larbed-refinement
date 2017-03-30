#ifndef YNPQNKONSBMXXGRXFFYJHNUQAWFFPQSMIMAFLFSCLXNACFFWOXXLFHQMEDQHYITQPOKWDMRIQ
#define YNPQNKONSBMXXGRXFFYJHNUQAWFFPQSMIMAFLFSCLXNACFFWOXXLFHQMEDQHYITQPOKWDMRIQ

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
    struct pattern
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

        //-------------------------------------------------------------------------------
        complex_type                            thickness;
        size_type                               ug_size;
        size_type                               tilt_size;
        size_type                               column_index;
        size_matrix_vector_type                 ar;         //ar[][][]
        matrix_vector_type                      diag;       //diag[][][]
        matrix_vector_type                      intensity;  //intensity[][][]
        complex_matrix_type                     ug;         //
        matrix_vector_type                      weights;  //weights[][][]

        void dump( std::string const& paht )
        {
        }

        void update_thickness( value_type val )
        {
            thickness = complex_type( 0.0, val );
        }
/*
        void add_noise( T const level = 0.01 )
        {
            variate_generator<T> vg( T{-1.0}, T{1.0} );
            for ( auto& mat : intensity )
                std::for_each( mat.begin(), mat.end(), [&vg,level]( value_type& x ) { x += vg() * level; });
        }
*/
        void renormalize_intensities()
        {
            for ( auto&& it : intensity )
            {
                //std::for_each( it.begin(), it.end(), [](value_type& x){ if ( x < value_type{0} ) x = value_type{0}; } );
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

        void update_ug( matrix<double> const& ugs )
        {
            assert( ugs.row() );
            assert( ugs.col() == 2 );
            ug.resize( ugs.row(), 1 );
            for ( unsigned long index = 0; index != ugs.row(); ++index )
                ug[index][0] = std::complex<double>{ ugs[index][0], ugs[index][1] };

            ug_size = ugs.row();
        }

        void update_diag( matrix<double> const& dia )
        {
            assert( dia.row() == diag[0].row() );

            unsigned long const size_to_update = std::min( diag.size(), dia.col() );

            for ( unsigned long index = 0; index != size_to_update; ++index )
                std::copy( dia.col_begin(index), dia.col_end(index), diag[index].begin() );

            simulate_intensity();
        }

        void simulate_intensity()
        {
            simulate_intensity( ug );
        }

        void simulate_intensity( complex_matrix_type const& ug_ )
        {
            assert( ug_.size() == ug_size );

            matrix_vector_type new_intensity;

            for ( size_type tilt_index = 0; tilt_index != tilt_size; ++tilt_index )
            {
                auto const& ar_ = ar[tilt_index];
                auto const& di_ = diag[tilt_index];
                auto const& s_mat = make_scattering_matrix( ar_, ug_, di_.begin(), di_.end(), thickness );

                matrix_type it{ s_mat.row(), 1 };
                std::transform( s_mat.col_begin(column_index), s_mat.col_end(column_index), it.begin(), []( complex_type const& c ){ return std::norm(c); } );
                new_intensity.push_back( std::move(it) );

            }

            new_intensity.swap( intensity );
        }
        /*
         *TODO: optimization herje
        void simulate_intensity( complex_matrix_type const& ug_, matrix_vector_type& new_intensity_ )
        {
            assert( ug_.size() == ug_size );
            new_intensity_.resize( tilt_size );

            for ( size_type tilt_index = 0; tilt_index != tilt_size; ++tilt_index )
            {
                auto const& ar_ = ar[tilt_index];
                auto const& di_ = diag[tilt_index];
                auto const& s_mat = make_scattering_matrix( ar_, ug_, di_.begin(), di_.end(), thickness );

                matrix_type it{ s_mat.row(), 1 };
                std::transform( s_mat.col_begin(column_index), s_mat.col_end(column_index), it.begin(), []( complex_type const& c ){ return std::norm(c); } );
                new_intensity_.push_back( std::move(it) );
            }
        }
        */

        friend std::ostream& operator << ( std::ostream& os, pattern<T> const& pt )
        {
            os << "thickness:\t"  << pt.thickness << "\n";
            os << "ug_size:\t"  << pt.ug_size << "\n";
            os << "tilt_size:\t"  << pt.tilt_size << "\n";
            os << "column_index:\t"  << pt.column_index << "\n";
            os << "ar size:\t"  << pt.ar.size() << "\n";
            os << "diag size:\t"  << pt.diag.size() << "\n";
            os << "intensity size:\t"  << pt.intensity.size() << "\n";
            os << "ug size:\t"  << pt.ug.size() << "\n";
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

    };//struct pattern

    static bool is_file_exist( std::string const& file_path )
    {
        std::ifstream ifs( file_path );
        return ifs.good();
    }

    template< typename T >
    pattern<T> const make_pattern( std::string const& dir_path_, std::complex<T> const& thickness_, std::size_t column_index_ = 0 )
    {
        typedef std::size_t size_type;
        typedef T value_type;
        typedef matrix<value_type> matrix_type;
        typedef matrix<size_type> size_matrix_type;
        typedef std::complex<value_type> complex_type;
        pattern<T> pt;
        pt.column_index = column_index_;

        if ( std::abs( std::real(thickness_) ) > T{1.0e-10} ) return pt;
        if ( std::imag(thickness_)  < T{1.0e-10} ) return pt;

        pt.thickness = thickness_;


        //load ug
        std::string const& ug_file_path = dir_path_ + std::string{"/_UgMasterList.txt"};
        std::ifstream ifs_ug{ ug_file_path };
        //if ( !ifs_ug.good() ) return pt;
        if ( ifs_ug.good() )
        {
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
        }

        {
            std::string const& ug_file_path = dir_path_ + std::string{"/ug.txt"};
            if ( is_file_exist( ug_file_path ) )
            {
                matrix<double> current_ug;
                current_ug.load( ug_file_path );
                assert( current_ug.row() && "Empty Ug file." );
                assert( current_ug.col() == 2 && "Ug real/complex term missing!" );
                pt.ug.resize( current_ug.row(), 1 );
                for ( auto r = 0UL; r != current_ug.row(); ++r )
                    pt.ug[r][0] = complex_type{ current_ug[r][0], current_ug[r][1] };
            }
        }

        size_matrix_type ar_x;
        matrix_type diag_x;
        matrix_type intensity_x;
        matrix_type weight_x;
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
            for ( auto& x : intensity_x )
            {
                if ( x < 0.0 )
                    x = 0.0;
            }

            pt.ar.push_back( ar_x );
            pt.diag.push_back( diag_x );
            pt.intensity.push_back( intensity_x );

            //load weights
            std::string const& weights_file_path = dir_path_ + std::string{"/Weights_"} + id + std::string{".txt"};
            if ( ! is_file_exist( weights_file_path) )
            {
                weight_x.resize( intensity_x.row(), intensity_x.col() );
                std::fill( weight_x.begin(), weight_x.end(), 1.0 );
            }
            else
            {
                weight_x.load( weights_file_path );
            }
            pt.weights.push_back( weight_x );
        }

        pt.ug_size = 0;
        for ( auto&& ar_ : pt.ar )
            pt.ug_size = std::max( pt.ug_size, *std::max_element(ar_.begin(), ar_.end()) );

        pt.ug_size++;

        pt.tilt_size = pt.diag.size();

        return pt;
    }

    template< typename T >
    pattern<T> const make_simulated_pattern( std::string const& dir_path_, std::complex<T> const& thickness_, std::size_t column_index_ = 0 )
    {
        auto pt = make_pattern( dir_path_, thickness_, column_index_ );
        pt.simulate_intensity();
        return pt;
    }

#if 0

    template< typename T >
    pattern<T> const make_pattern_n( unsigned long const max_pattern, std::string const& dir_path_, std::complex<T> const& thickness_, std::size_t column_index_ = 0 )
    {
        typedef std::size_t size_type;
        typedef T value_type;
        typedef matrix<value_type> matrix_type;
        typedef matrix<size_type> size_matrix_type;
        typedef std::complex<value_type> complex_type;
        pattern<T> pt;
        pt.column_index = column_index_;

        if ( std::abs( std::real(thickness_) ) > T{1.0e-10} ) return pt;
        if ( std::imag(thickness_)  < T{1.0e-10} ) return pt;

        pt.thickness = thickness_;

        //load ug
        std::string const& ug_file_path = dir_path_ + std::string{"/_UgMasterList.txt"};
        std::ifstream ifs_ug{ ug_file_path };
        //if ( !ifs_ug.good() ) return pt;
        if ( ifs_ug.good() )
        {
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
        }

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
            for ( auto& x : intensity_x )
            {
                if ( x < 0.0 )
                    x = 0.0;
            }

            pt.ar.push_back( ar_x );
            pt.diag.push_back( diag_x );
            pt.intensity.push_back( intensity_x );


            if ( max_pattern == index + 1 ) break;
        }

        pt.ug_size = 0;
        for ( auto&& ar_ : pt.ar )
            pt.ug_size = std::max( pt.ug_size, *std::max_element(ar_.begin(), ar_.end()) );

        pt.ug_size++;

        pt.tilt_size = pt.diag.size();

        return pt;
    }

    template< typename T >
    std::map<unsigned long, T> const extract_ug_norm( pattern<T> const& pt )
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
            auto coef = coefficient<value_type>{ pt.thickness, (pt.diag)[index].begin(), (pt.diag)[index].end() };
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
            //value_type const the_norm = std::sqrt( sum_weighted_ug / sum_weigh );
            value_type const the_norm = sum_weighted_ug / sum_weigh;
            ug_norm[ug_index] = the_norm;
        }

        return ug_norm;
    }
#endif
/*
    template< typename T, typename Cond >
    pattern<T> const make_pattern( Cond cond_, std::string const& dir_path_, std::complex<T> const& thickness_, std::size_t column_index_ = 0 )
    {
        typedef std::size_t size_type;
        typedef T value_type;
        typedef matrix<value_type> matrix_type;
        typedef matrix<size_type> size_matrix_type;
        typedef std::complex<value_type> complex_type;
        pattern<T> pt;
        pt.column_index = column_index_;

        if ( std::abs( std::real(thickness_) ) > T{1.0e-10} ) return pt;
        if ( std::imag(thickness_)  < T{1.0e-10} ) return pt;

        pt.thickness = thickness_;

        //load ug
        std::string const& ug_file_path = dir_path_ + std::string{"/_UgMasterList.txt"};
        std::ifstream ifs_ug{ ug_file_path };
        //if ( !ifs_ug.good() ) return pt;

        if ( ifs_ug.good() )
        {
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
        }

        size_matrix_type ar_x;
        matrix_type diag_x;
        matrix_type intensity_x;
        for ( size_type index = 0; true; ++index )
        {
            std::string const& id = std::to_string( index );
            std::string const& ar_file_path = dir_path_ + std::string{"/Ar_"} + id + std::string{".txt"};
            std::string const& diag_file_path = dir_path_ + std::string{"/Diag_"} + id + std::string{".txt"};
            std::string const& intensity_file_path = dir_path_ + std::string{"/Intensities_"} + id + std::string{".txt"};

            if ( ! ( is_file_exist( ar_file_path ) && is_file_exist( diag_file_path ) && is_file_exist( intensity_file_path ) ) )
            {
                break;
            }
            if ( !cond_(index) ) continue;

            ar_x.load( ar_file_path );
            diag_x.load( diag_file_path );
            intensity_x.load( intensity_file_path );
            for ( auto& x : intensity_x )
            {
                if ( x < 0.0 )
                    x = 0.0;
            }

            std::for_each( intensity_x.begin(), intensity_x.end(), [](value_type& x){ x = std::max(x, value_type{0}); } );
            value_type sum = std::accumulate( intensity_x.begin(), intensity_x.end(), value_type{0} );
            std::for_each( intensity_x.begin(), intensity_x.end(), [sum](value_type& x){ x /= sum; } );

            pt.ar.push_back( ar_x );
            pt.diag.push_back( diag_x );
            pt.intensity.push_back( intensity_x );
        }

        pt.ug_size = 0;
        for ( auto&& ar_ : pt.ar )
            pt.ug_size = std::max( pt.ug_size, *std::max_element(ar_.begin(), ar_.end()) );

        pt.ug_size++;

        pt.tilt_size = pt.diag.size();

        return pt;
    }
*/

}//namespace f

#endif//YNPQNKONSBMXXGRXFFYJHNUQAWFFPQSMIMAFLFSCLXNACFFWOXXLFHQMEDQHYITQPOKWDMRIQ

