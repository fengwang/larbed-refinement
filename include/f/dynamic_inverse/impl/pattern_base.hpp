#ifndef OYYCQILRUDRLJBPIXPAQEJLHERIUBUYNAUKETWJQHLTUCCJGDMUPYKGBYEQMONIDUFROYMXRM
#define OYYCQILRUDRLJBPIXPAQEJLHERIUBUYNAUKETWJQHLTUCCJGDMUPYKGBYEQMONIDUFROYMXRM

#include <f/matrix/matrix.hpp>

#include <cstddef>
#include <complex>
#include <string>

namespace f
{

    template<typename T, typename Zen>
    struct pattern_base //mono
    {
        typedef Zen                                         zen_type;
        typedef T                                           value_type;
        typedef matrix<T>                                   matrix_type;
        typedef std::size_t                                 size_type;
        typedef matrix<size_type>                           size_matrix_type;
        typedef std::string                                 string_type;
        typedef typename matrix_type::const_col_type        const_col_type;

        //just for static sized ar here
        size_type                                           dimension_;
        size_type                                           total_tilt_;
        size_matrix_type                                    ar_;
        matrix_type                                         intensity_;
        matrix_type                                         diag_;

        void dump_pattern_base() const
        {
            std::cout << "\npattern_base_dump:\n";
            std::cout << "\ndimension is \t" << dimension_ << "\n";
            std::cout << "\nAr is " << ar_.row() << " X " << ar_.col() << "\n";
            std::cout << "\nIntensity is " << intensity_.row() << " X " << intensity_.col() << "\n";
            std::cout << "\nDiag is " << diag_.row() << " X " << diag_.col() << "\n";
            std::cout << std::endl;
        }

        void config_total_tilt( size_type total_tilt__ )
        {
            total_tilt_ = total_tilt__;
        }

        size_type total_tilt() const
        {
            return total_tilt_;
        }

        size_type dimension( size_type /*tilt_index*/ ) const
        {
            return dimension_;
        }

        const_col_type diag_begin( size_type index ) const
        {
            //asserts
            return diag_.col_begin( index );
        }

        const_col_type diag_end( size_type index ) const
        {
            //asserts
            return diag_.col_end( index );
        }

        size_matrix_type const ar( size_type /*index*/ ) const
        {
            return ar_;
        }

        size_type ar( size_type /*index*/, size_type row, size_type col ) const
        {
            return ar_[row][col];
        }

#if 0
        matrix_type const diag( size_type index ) const
        {
            //return a col matrix
            return matrix_type{diag, 0, diag.row(), index, index+1};
        }
#endif

        value_type const intensity( size_type const pattern_index, size_type const tilt_index ) const
        {
            return intensity_[pattern_index][tilt_index];
        }
/*
        value_type const intensity( size_type index_of_pattern, size_type offset  ) const
        {
            //asserts
            return intensity_[offset][index_of_pattern];
        }
        */

        void config_dimension( size_type /*index*/, size_type dimension )
        {
            dimension_ = dimension;
        }

        void config_ar( size_type /*index*/, size_matrix_type const& ar__ )
        {
            ar_ = ar__;
        }

        matrix_type const intensity( size_type index ) const
        {
            return matrix_type{ intensity_, 0, intensity_.row(), index, index+1 };
        }

        void config_intensity( size_type index_of_pattern, matrix_type const& intensity )
        {
            //asserts
            std::copy( intensity.begin(), intensity.end(), intensity_.col_begin(index_of_pattern) );
        }

        value_type const diag( size_type index_of_pattern, size_type offset  ) const
        {
            //asserts
            return diag_[offset][index_of_pattern];
        }


        void config_diag( size_type index_of_pattern, matrix_type const& diag )
        {
            //asserts
            std::copy( diag.begin(), diag.end(), diag_.col_begin(index_of_pattern) );
        }

        void load( string_type const& path )
        {
            //we might want to overload these methods in later implementations
            zen_type& zen = static_cast<zen_type&>(*this);
            zen.load_ar( path );
            zen.load_intensity( path );
            zen.load_diag( path );
        }

        void load_ar( string_type const& path )
        {
            size_matrix_type ar_tmp{ path + string_type{"/Ar.txt"} };
            ar_tmp.swap( ar_ );
        }

        void load_intensity( string_type const& path )
        {
            intensity_.resize( dimension_, total_tilt_ );
            matrix_type intensity_tmp{ dimension_+1, 1 }; //SSTO, the first element is a counter
            for ( size_type c = 0; c != total_tilt_; ++c )
            {
                intensity_tmp.load( path + string_type{"/Intensities_"} + std::to_string(c) + string_type{".txt"} );
                std::copy( intensity_tmp.col_begin(0)+1, intensity_tmp.col_end(0), intensity_.col_begin(c) );
            }
        }

        void load_diag( string_type const& path )
        {
            diag_.resize( dimension_, total_tilt_ );
            matrix_type diag_tmp{ dimension_, 1 };
            for ( size_type c = 0; c != total_tilt_; ++c )
            {
                diag_tmp.load( path + string_type{"/Diag_"} + std::to_string(c) + string_type{".txt"} );
                std::copy( diag_tmp.col_begin(0), diag_tmp.col_end(0), diag_.col_begin(c) );
            }
        }

    };//struct pattern_base

    template<typename T, typename Zen>
    struct selected_pattern_base //mono
    {
    };//struct selected_pattern_base

}//namespace f

#endif//OYYCQILRUDRLJBPIXPAQEJLHERIUBUYNAUKETWJQHLTUCCJGDMUPYKGBYEQMONIDUFROYMXRM

