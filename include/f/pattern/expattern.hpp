#ifndef NVLOCQMFRXKVJGCFFSQNSEGYUIELEHBTLRQCKWHYBSGXOQHWUWJRTUETKSWYFOXSWROKRLWHX
#define NVLOCQMFRXKVJGCFFSQNSEGYUIELEHBTLRQCKWHYBSGXOQHWUWJRTUETKSWYFOXSWROKRLWHX


#include <f/matrix/matrix.hpp>
#include <f/lexical_cast/lexical_cast.hpp>
#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <cstddef>
#include <string>
#include <complex>

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

        //-------------------------------------------------------------------------------
        complex_type                            thickness;
        size_type                               ug_size;
        size_type                               tilt_size;
        size_type                               column_index;
        size_matrix_vector_type                 ar;         //ar[][][]
        matrix_vector_type                      diag;       //diag[][][]
        matrix_vector_type                      intensity;  //intensity[][][]
        matrix_vector_type                      simulated_intensity;  //intensity[][][]
        complex_matrix_type                     ug;         //
        matrix_type                             tilt_angle; //stores the cosine

        void load_intensity( std::string const& path )
        {
            matrix_type     raw_intensity;
            raw_intensity.load( path );

            assert( raw_intensity.row() == ug_size );
            assert( raw_intensity.col() == tilt_size );

            for ( size_type tilt_index = 0; tilt_index != tilt_size; ++tilt_index )
            {
                intensity[tilt_index].resize( ar[tilt_index].row(), 1 );
                for ( size_type row = 0; row != ar[tilt_index].row(); ++row )
                {
                    size_type const ug_index = ar[tilt_index][row][column_index];
                    value_type const intensity_rc = raw_intensity[ug_index][tilt_index];
                    intensity[tilt_index][row][0] = intensity_rc;
                }
            }
        }

        value_type make_diff_2( complex_matrix_type const& ug_ )
        {
            simulate_intensity( ug_ );

            value_type diff_2 = value_type{0};

            for ( size_type tilt_index = 0; tilt_index != tilt_size; ++tilt_index )        
            {
                matrix_type const& diff_mat = simulated_intensity[tilt_index] - intensity[tilt_index];
                diff_2 += std::inner_product( diff_mat.begin(), diff_mat.end(), diff_mat.begin(), value_type{0} );
            }

            return diff_2;
        }

        void set_thickness( complex_type const& thickness_ )
        {
            assert( std::abs(std::real(thickness_)) < value_type{1.0e-10} );
            assert( td::imag(thickness_) > value_type{1.0e-10} );
            thickness = thickness_;
        }

        value_type make_diff_2()
        {
            return make_diff_2( (*this).ug );
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
                //std::for_each( it.begin(), it.end(), [](value_type& x){ if ( x < value_type{0} ) x = value_type{0}; } );
                value_type sum_up = std::accumulate( it.begin(), it.end(), value_type{0});
                std::for_each( it.begin(), it.end(), [sum_up](value_type& x){ x /= sum_up; } );
            }
        }

        /*
        void normalize_intensities()
        {
            for ( auto&& it : intensity )
            {
                std::for_each( it.begin(), it.end(), [](value_type& x){ if ( x < value_type{0} ) x = value_type{0}; } );
                value_type sum_up = std::accumulate( it.begin(), it.end(), value_type{0});
                std::for_each( it.begin(), it.end(), [sum_up](value_type& x){ x /= sum_up; } );
            }
        }
        */

        void simulate_intensity()
        {
            simulate_intensity( ug );
        }

        void simulate_intensity( complex_matrix_type const& ug_ )
        {
            //std::cerr << ug_.size() << "\n";
            //std::cerr << ug_size << "\n";
            assert( ug_.size() == ug_size ); 
            matrix_vector_type new_intensity;

            for ( size_type tilt_index = 0; tilt_index != tilt_size; ++tilt_index )
            {
                auto const& ar_ = ar[tilt_index];
                auto const& di_ = diag[tilt_index];
                auto const& s_mat = make_scattering_matrix( ar_, ug_, di_.begin(), di_.end(), thickness/tilt_angle[tilt_index][0] );

                matrix_type it{ s_mat.row(), 1 };
                std::transform( s_mat.col_begin(column_index), s_mat.col_end(column_index), it.begin(), []( complex_type const& c ){ return std::norm(c); } );
                new_intensity.push_back( std::move(it) );
            }

            //new_intensity.swap( intensity );
            new_intensity.swap( simulated_intensity );
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

        friend std::ostream& operator << ( std::ostream& os, xpattern<T> const& pt )
        {
            /*
            os << "thickness:\t"  << pt.thickness << "\n";
            os << "ug_size:\t"  << pt.ug_size << "\n";
            os << "tilt_size:\t"  << pt.tilt_size << "\n";
            os << "column_index:\t"  << pt.column_index << "\n";
            os << "ar:\n"  << pt.ar[0] << "\n";

            os << "diag\n";
            for ( std::size_t i = 0; i != pt.tilt_size; ++i )
                os << pt.diag[i].transpose() << "\n";
            */

            os << "intensity\n";
            for ( std::size_t i = 0; i != pt.tilt_size; ++i )
                os << pt.intensity[i].transpose() << "\n";

            /*
            os << "ug:\n"  << pt.ug << "\n";
            os << "theta:\n"  << pt.tilt_angle << "\n";
            */

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

        //load tilt angles
        matrix_type tilts_mat;
        tilts_mat.load( dir_path_ + "/_Tilts.txt" );
        pt.tilt_angle.resize( tilts_mat.row(), 1 );
        for ( size_type r = 0; r != pt.tilt_angle.row(); ++r )
        {
            const value_type theta_1 = std::sin( tilts_mat[r][4] / value_type{1000} );
            const value_type theta_2 = std::sin( tilts_mat[r][5] / value_type{1000} );
            pt.tilt_angle[r][0] = std::sqrt( value_type{1} - theta_1 * theta_1 - theta_2 * theta_2 );
        }

        return pt;
    }

    template< typename T >
    xpattern<T> const make_simulated_xpattern( std::string const& dir_path_, std::complex<T> const& thickness_, std::size_t column_index_ = 0 )
    {
        auto pt = make_xpattern( dir_path_, thickness_, column_index_ );
        pt.simulate_intensity();
        return pt;
    }

}//namespace f

#endif//NVLOCQMFRXKVJGCFFSQNSEGYUIELEHBTLRQCKWHYBSGXOQHWUWJRTUETKSWYFOXSWROKRLWHX

