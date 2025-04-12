/**
 * @file   BookshelfDataBase.h
 * @brief  Database for Bookshelf parser 
 * @author Rachel Selina
 * @date   Jan 2021
 */

#ifndef BOOKSHELFPARSER_DATABASE_H
#define BOOKSHELFPARSER_DATABASE_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <limits>

/// namespace for BookshelfParser
namespace BookshelfParser {

/// @nowarn
using std::cout;
using std::endl;
using std::cerr;
using std::string; 
using std::vector;
using std::pair;
using std::make_pair;
using std::ostream;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
/// @endnowarn

/// @brief bison does not support vector very well, 
/// so here create a dummy class for integer array. 
//class IntegerArray : public vector<int>
//{
//	public: 
//        /// @nowarn 
//		typedef vector<int> base_type;
//		using base_type::size_type;
//		using base_type::value_type;
//		using base_type::allocator_type;
//        /// @endnowarn
//
//        /// constructor 
//        /// @param alloc memory allocator 
//		IntegerArray(const allocator_type& alloc = allocator_type())
//			: base_type(alloc) {}
//        /// constructor 
//        /// @param n number of values 
//        /// @param val data value
//        /// @param alloc memory allocator 
//		IntegerArray(size_type n, const value_type& val, const allocator_type& alloc = allocator_type())
//			: base_type(n, val, alloc) {}
//};

/// @brief bison does not support vector very well, 
/// so here create a dummy class for string array. 
class StringArray : public vector<string>
{
	public: 
        /// @nowarn 
		typedef vector<string> base_type;
		using base_type::size_type;
		using base_type::value_type;
		using base_type::allocator_type;
        /// @endnowarn

        /// constructor 
        /// @param alloc memory allocator 
		StringArray(const allocator_type& alloc = allocator_type())
			: base_type(alloc) {}
        /// constructor 
        /// @param n number of values 
        /// @param val data value
        /// @param alloc memory allocator 
		StringArray(size_type n, const value_type& val, const allocator_type& alloc = allocator_type())
			: base_type(n, val, alloc) {}
};

/// @brief Temporary data structures to hold parsed data. 
/// Base class for all temporary data structures. 
struct Item 
{
    /// print data members 
	virtual void print(ostream&) const {};
    /// print data members with stream operator 
    /// @param ss output stream 
    /// @param rhs target object 
    /// @return output stream 
	friend ostream& operator<<(ostream& ss, Item const& rhs)
	{
		rhs.print(ss);
		return ss;
	}
};
/// @brief describe a pin of a net 
struct NetPin : public Item 
{
    string node_name; ///< node name 
    string pin_name; ///< pin name 
    //char direct; ///< direction 
    //double offset[2]; ///< offset (x, y) to node origin 
    //double size[2]; ///< sizes (x, y) of pin 

    /// constructor 
    NetPin()
    {
        node_name = "";
        pin_name = ""; 
        //direct = '\0';
        //offset[0] = 0; 
        //offset[1] = 0; 
        //size[0] = 0;
        //size[1] = 0;
    }
    /// constructor 
    /// @param nn node name 
    /// @param pn pin name 
    NetPin(string& nn, string& pn)
    {
        node_name.swap(nn);
        pin_name.swap(pn);
    }
   /// constructor 
    /// @param nn node name 
    /// @param d direction 
    /// @param x, y offset of pin to node origin 
    /// @param w, h size of pin 
    /// @param pn pin name 
    //NetPin(string& nn, char d, double x, double y, double w, double h, string& pn)
    //{
    //    node_name.swap(nn);
    //    direct = d;
    //    offset[0] = x;
    //    offset[1] = y;
    //    size[0] = w;
    //    size[1] = h;
    //    pin_name.swap(pn);
    //}
    /// constructor 
    /// @param nn node name 
    /// @param d direction 
    /// @param x, y offset of pin to node origin 
    /// @param w, h size of pin 
    //NetPin(string& nn, char d, double x, double y, double w, double h)
    //{
    //    node_name.swap(nn);
    //    direct = d;
    //    offset[0] = x;
    //    offset[1] = y;
    //    size[0] = w;
    //    size[1] = h;
    //    pin_name.clear();
    //}
    /// reset all data members 
    void reset()
    {
        node_name = "";
        pin_name = "";
        //direct = 'I';
        //offset[0] = offset[1] = 0;
        //size[0] = size[1] = 0;
    }
};
/// @brief net to describe interconnection of netlist 
struct Net : public Item
{
	string net_name; ///< net name 
	vector<NetPin> vNetPin; ///< array of pins in the net 
    /// reset all data members 
	void reset()
	{
		net_name = "";
		vNetPin.clear();
	}
    /// print data members 
    /// @param ss output stream 
	virtual void print(ostream& ss) const
	{
		ss << "//////// Net ////////" << endl
			<< "net_name = " << net_name << endl; 
		for (uint32_t i = 0; i < vNetPin.size(); ++i)
			ss << "(" << vNetPin[i].node_name << ", " << vNetPin[i].pin_name << ") "; 
                //<< vNetPin[i].direct << " @(" << vNetPin[i].offset[0] << ", " << vNetPin[i].offset[1] << ")";
		ss << endl;
	}
};
/// @brief carrychain to describe carry chains in design
struct CarryChain: public Item
{
	string name; ///< name of carry chain
    int elCount; ///< number of nodes in carry chain 
	vector<std::string> elements; ///< array of elements in the carry chain 
    /// reset all data members 
	void reset()
	{
		name = "";
        elCount = 0;
		elements.clear();
	}
    /// print data members 
    /// @param ss output stream 
	virtual void print(ostream& ss) const
	{
		ss << "//////// CarryChain ////////" << endl
			<< "name = " << name << " contains " << elCount << " nodes " << endl; 
		for (uint32_t i = 0; i < elements.size(); ++i)
			ss << elements[i] << " "; 
		ss << endl;
	}
};

/// @brief site to describe resources and their count
struct Site : public Item
{
	string name; ///< site name 
	vector<std::pair<std::string, int> > rsrcs; ///< array of resources and count within each site
    /// reset all data members 
	void reset()
	{
		name = "";
		rsrcs.clear();
	}
    /// print data members 
    /// @param ss output stream 
	virtual void print(ostream& ss) const
	{
		ss << "//////// Site ////////" << endl
			<< "name = " << name << endl; 
		for (uint32_t i = 0; i < rsrcs.size(); ++i)
			ss << "(" << rsrcs[i].first << " : " << rsrcs[i].second << ") "; 
		ss << endl;
	}
};
/// @brief rsrc to describe instances in resource
struct Rsrc : public Item
{
	string name; ///< resource name 
	vector<std::string> rsrcCells; ///< array of resource instances within each resource type
    /// reset all data members 
	void reset()
	{
		name = "";
		rsrcCells.clear();
	}
    /// print data members 
    /// @param ss output stream 
	virtual void print(ostream& ss) const
	{
		ss << "//////// Rsrc ////////" << endl
			<< "name = " << name << endl; 
		for (uint32_t i = 0; i < rsrcCells.size(); ++i)
			ss << rsrcCells[i] << " "; 
		ss << endl;
	}
};
/// @brief LUTFract to describe instances in LUT fracture 
struct LUTFract : public Item
{
	string name; ///< LUT name 
	vector<std::string> fractCells; ///< array of fracture LUT combinations
    /// reset all data members 
	void reset()
	{
		name = "";
		fractCells.clear();
	}
    /// print data members 
    /// @param ss output stream 
	virtual void print(ostream& ss) const
	{
		ss << "//////// LUTFract ////////" << endl
			<< "name = " << name << endl; 
		for (uint32_t i = 0; i < fractCells.size(); ++i)
			ss << fractCells[i] << " "; 
		ss << endl;
	}
};
/// @brief SiteOut to describe placement output format
struct SiteOut : public Item
{
	string coordinate; ///< coordinate
	int value; ///< value
	vector<std::string> siteTypes; ///< array of site types
    /// reset all data members
	void reset()
	{
		coordinate = "";
        value = 0;
		siteTypes.clear();
	}
    /// print data members
    /// @param ss output stream
	virtual void print(ostream& ss) const
	{
		ss << "//////// SiteOut ////////" << endl
			<< "coordinate = " << coordinate << ", value = " << value << endl; 
		for (uint32_t i = 0; i < siteTypes.size(); ++i)
			ss << siteTypes[i] << " "; 
		ss << endl;
	}
};

// forward declaration
/// @brief Base class for bookshelf database. 
/// Only pure virtual functions are defined.  
/// User needs to inheritate this class and derive a custom database type with all callback functions defined.  
class BookshelfDataBase
{
	public:
        /// @brief add node 
        //virtual void add_bookshelf_node(string&, int, int) = 0;
        virtual void add_bookshelf_node(string&, string&) = 0;
        /// @brief add net 
        virtual void add_bookshelf_net(Net const&) = 0;
        /// @brief add carry 
        virtual void add_bookshelf_carry(CarryChain const&) = 0;
        /// @brief a callback to update nodes 
        virtual void update_nodes() = 0;
        /// @brief set node position 
        virtual void set_bookshelf_node_pos(string const&, double, double, int) = 0;
        ///@brief update site size 
        virtual void resize_sites(int, int) = 0;
        ///@brief site 
        virtual void add_site(Site const&) = 0;
        ///@brief rsrc
        virtual void add_rsrc(Rsrc const&) = 0;
        ///@brief update site information
        virtual void site_info_update(int, int, string const&) = 0;
        ///@brief number of clock regions
        virtual void resize_clk_regions(int, int) = 0;
        ///@brief update clock regions
        virtual void add_clk_region(string const&, int, int, int, int, int, int) = 0;
        /// @brief add lib cell
        virtual void add_lib_cell(string const&) = 0; 
        /// @brief add lib cell input pin
        virtual void add_input_pin(string&) = 0; 
        /// @brief add lib cell output pin
        virtual void add_output_pin(string&) = 0; 
        /// @brief add lib cell output adder pin
        virtual void add_output_add_pin(string&) = 0; 
        /// @brief add lib cell clk pin
        virtual void add_clk_pin(string&) = 0; 
        /// @brief add lib cell ctrl pin
        virtual void add_ctrl_pin(string&) = 0; 
        /// @brief add lib cell input adder pin
        virtual void add_input_add_pin(string&) = 0; 
        /// @brief set site per column 
        virtual void set_site_per_column(int) = 0; 
        /// @brief set site dimensions
        virtual void set_site_dimensions(const string&, double, double) = 0; 
        /// @brief set slice element
        virtual void set_slice_element(const string&, int) = 0; 
        /// @brief set cell dimensions
        virtual void set_cell_dimensions(const string&, double, double) = 0; 
        /// @brief set lut max shared
        virtual void set_lut_max_shared(int) = 0; 
        /// @brief set lut type in sliceUnit
        virtual void set_lut_type_in_sliceUnit(int) = 0; 
        ///@brief lutfract 
        virtual void set_lut_fractureability(LUTFract const&) = 0;
        /// @brief set ff slice ctrl mode
        virtual void set_sliceFF_ctrl_mode(const string&) = 0; 
        /// @brief set ff slice ctrl signals 
        virtual void set_sliceFF_ctrl(const string&, int) = 0; 
        /// @brief set ff slice unit ctrl signals 
        virtual void set_sliceUnitFF_ctrl(const string&, int) = 0; 
        /// @brief set ff ctrl type 
        virtual void set_FFCtrl_type(const string&) = 0; 
        /// @brief set wl weight
        virtual void set_wl_weight_x(double) = 0; 
        virtual void set_wl_weight_y(double) = 0; 
        /// @brief set route cap
        virtual void set_pin_route_cap(int) = 0; 
        virtual void set_route_cap_H(int) = 0; 
        virtual void set_route_cap_V(int) = 0; 
        /// @brief set site out
        virtual void set_siteOut(SiteOut const&) = 0; 
        //
        /// @brief set design name 
        virtual void set_bookshelf_design(string&) = 0;
        /// @brief a callback when a bookshelf file reaches to the end 
        virtual void bookshelf_end() = 0;
    private:
        /// @brief remind users to define some optional callback functions at runtime 
        /// @param str message including the information to the callback function in the reminder 
        void bookshelf_user_cbk_reminder(const char* str) const;
};

} // namespace BookshelfParser

#endif
